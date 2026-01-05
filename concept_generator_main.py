# pyright: reportMissingImports=false
"""LangChain-powered multi-agent system for UMLS concept discovery.

This module exposes a high-level orchestration layer that coordinates multiple
LangChain agents to search the Unified Medical Language System (UMLS) for a
given disease, collect relevant concepts, and expand them with related
terminology. It replaces the previous direct REST calls with a more flexible
agentic workflow.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Type

import requests
from pydantic import BaseModel, Field
from requests import Response, Session

try:
    # Try newer import path first (langchain-core >= 1.0)
    try:
        from langchain_core.tools import BaseTool
    except ImportError:
        # Fallback to older import path
        from langchain.tools import BaseTool
    
    from langchain_openai import ChatOpenAI
    from langchain_experimental.plan_and_execute import (
        PlanAndExecute,
        load_agent_executor,
        load_chat_planner,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "LangChain dependencies are missing. Install the packages listed in "
        "`requirements.txt` (langchain, langchain-openai, langchain-experimental, pydantic) before using this module."
    ) from exc

LOGGER = logging.getLogger(__name__)

UMLS_API_BASE = "https://uts-ws.nlm.nih.gov/rest"
DEFAULT_KEY_PATH = Path(__file__).resolve().parent / ".umls_api_key"
DEFAULT_OPENAI_KEY_PATH = Path(__file__).resolve().parent / ".openai_api_key"
# Note: Since May 2022, UMLS API supports direct API key authentication.
# The old ticket-granting ticket system is deprecated.


class UMLSClient:
    """Lightweight client that manages UMLS authentication and requests.
    
    Uses direct API key authentication (supported since May 2022).
    See: https://documentation.uts.nlm.nih.gov/rest/home.html
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        session: Optional[Session] = None,
        api_base: str = UMLS_API_BASE,
        request_timeout: int = 30,
    ) -> None:
        key = api_key or self._read_key_from_disk(DEFAULT_KEY_PATH)
        if not key:
            raise ValueError(
                "A UMLS API key is required. Provide one explicitly or store it in "
                f"{DEFAULT_KEY_PATH}"
            )

        self.api_key = key.strip()
        self.session = session or Session()
        self.api_base = api_base.rstrip("/")
        self.request_timeout = request_timeout

    @staticmethod
    def _read_key_from_disk(path: Path) -> Optional[str]:
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            LOGGER.debug("UMLS API key file not found at %s", path)
        return None

    def _authenticated_get(
        self, path: str, *, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make an authenticated GET request using direct API key authentication."""
        query = dict(params or {})
        query["apiKey"] = self.api_key
        url = f"{self.api_base}/{path.lstrip('/')}"
        LOGGER.debug("GET %s (apiKey included)", url)
        response = self.session.get(url, params=query, timeout=self.request_timeout)
        self._raise_for_status(response, f"UMLS GET request to {url} failed")
        return response.json()

    def _authenticated_post(
        self,
        path: str,
        *,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make an authenticated POST request using direct API key authentication."""
        query = dict(params or {})
        query["apiKey"] = self.api_key
        url = f"{self.api_base}/{path.lstrip('/')}"
        LOGGER.debug("POST %s (apiKey included)", url)
        response = self.session.post(
            url, params=query, data=data, timeout=self.request_timeout
        )
        self._raise_for_status(response, f"UMLS POST request to {url} failed")
        return response.json()

    @staticmethod
    def _raise_for_status(response: Response, message: str) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            raise RuntimeError(f"{message}: {detail}") from exc

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def search_concepts(
        self,
        query: str,
        *,
        page_size: int = 10,
        search_type: str = "words",
        include_obsolete: bool = False,
        include_suppressible: bool = False,
    ) -> List[Dict[str, Any]]:
        """Search for UMLS concepts matching the query string."""
        params = {
            "string": query,
            "pageSize": page_size,
            "searchType": search_type,
            "includeObsolete": str(include_obsolete).lower(),
            "includeSuppressible": str(include_suppressible).lower(),
        }
        payload = self._authenticated_get("search/current", params=params)
        results = payload.get("result", {}).get("results", [])
        LOGGER.debug("Search returned %d results for %s", len(results), query)
        return results

    def fetch_concept_details(
        self, cui: str, *, return_format: str = "json"
    ) -> Dict[str, Any]:
        """Fetch detailed information for a concept identified by its CUI."""
        path = f"content/current/CUI/{cui}"
        params = {"returnIdType": "sourceConcept", "format": return_format}
        payload = self._authenticated_get(path, params=params)
        result = payload.get("result", payload)
        # Ensure we always return a dict
        if isinstance(result, dict):
            return result
        elif isinstance(result, str):
            # If result is a string, return empty dict
            LOGGER.warning("Unexpected string result from fetch_concept_details for CUI %s", cui)
            return {}
        else:
            # Fallback to empty dict for any other type
            return {}

    def fetch_related_concepts(
        self,
        cui: str,
        *,
        relation_types: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve concepts related to the given CUI."""
        params: Dict[str, Any] = {}
        if relation_types:
            params["includedRelation"] = ",".join(sorted(set(relation_types)))
        path = f"content/current/CUI/{cui}/relations"
        payload = self._authenticated_get(path, params=params)
        results = payload.get("result", [])
        LOGGER.debug(
            "Relations request returned %d items for CUI %s", len(results), cui
        )
        return results


# ------------------------------------------------------------------------- #
# LangChain tool definitions
# ------------------------------------------------------------------------- #


class SearchUMLSInput(BaseModel):
    query: str = Field(..., description="Disease or keyword to search in UMLS.")
    page_size: int = Field(
        10, ge=1, le=100, description="Maximum number of search results to return."
    )
    search_type: str = Field(
        "words",
        description="UMLS searchType parameter (e.g., 'exact', 'words', 'approximate').",
    )


class SearchUMLSTool(BaseTool):
    name: str = "search_umls_concepts"
    description: str = (
        "Use this tool to search the UMLS Metathesaurus for concepts related to a "
        "disease, symptom, or biomedical term. Returns JSON with 'count' (number of results), "
        "'results' (array of concepts), and 'message' fields. Each result has 'ui' (CUI), "
        "'name', and 'score'. ALWAYS check 'count' field first - if count is 0, the results "
        "array is empty. Only access results[0], results[1], etc. if count > 0."
    )
    args_schema: Type[BaseModel] = SearchUMLSInput

    def __init__(self, client: UMLSClient, **kwargs: Any):
        super().__init__(**kwargs)
        object.__setattr__(self, "client", client)

    def _run(  # type: ignore[override]
        self, query: str, page_size: int = 10, search_type: str = "words"
    ) -> str:
        results = self.client.search_concepts(
            query, page_size=page_size, search_type=search_type
        )
        # Wrap response to make empty results explicit
        response = {
            "count": len(results),
            "results": results,
            "message": f"Found {len(results)} result(s) for query '{query}'"
        }
        return json.dumps(response, ensure_ascii=False)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover
        raise NotImplementedError("Async operation is not supported.")


class ConceptDetailInput(BaseModel):
    cui: str = Field(..., description="UMLS Concept Unique Identifier (CUI).")


class FetchConceptDetailsTool(BaseTool):
    name: str = "fetch_umls_concept_details"
    description: str = (
        "Retrieve detailed information for a specific UMLS concept by its CUI. Use "
        "this after searching to gather definitions, semantic types, and source data."
    )
    args_schema: Type[BaseModel] = ConceptDetailInput

    def __init__(self, client: UMLSClient, **kwargs: Any):
        super().__init__(**kwargs)
        object.__setattr__(self, "client", client)

    def _run(self, cui: str) -> str:  # type: ignore[override]
        details = self.client.fetch_concept_details(cui)
        # Wrap response to ensure consistent structure
        if not details:
            return json.dumps({"error": f"No details found for CUI {cui}", "cui": cui}, ensure_ascii=False)
        return json.dumps(details, ensure_ascii=False)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover
        raise NotImplementedError("Async operation is not supported.")


class RelatedConceptsInput(BaseModel):
    cui: str = Field(..., description="UMLS Concept Unique Identifier (CUI).")
    relation_types: Optional[List[str]] = Field(
        default=None,
        description="Optional list of relation types to include (e.g., 'RO', 'RB').",
    )


class FetchRelatedConceptsTool(BaseTool):
    name: str = "fetch_umls_related_concepts"
    description: str = (
        "Fetch concepts related to a specific UMLS CUI. Use this to expand a disease "
        "concept into associated findings, treatments, anatomical sites, etc. Returns JSON "
        "with 'count' (number of relations), 'relations' (array), and 'message' fields. "
        "ALWAYS check 'count' field first - if count is 0, the relations array is empty. "
        "Each relation has 'relatedId' (CUI), 'relatedIdName', 'relationLabel', etc."
    )
    args_schema: Type[BaseModel] = RelatedConceptsInput

    def __init__(self, client: UMLSClient, **kwargs: Any):
        super().__init__(**kwargs)
        object.__setattr__(self, "client", client)

    def _run(  # type: ignore[override]
        self, cui: str, relation_types: Optional[List[str]] = None
    ) -> str:
        relations = self.client.fetch_related_concepts(cui, relation_types=relation_types)
        # Wrap response to make empty results explicit
        response = {
            "count": len(relations),
            "relations": relations,
            "message": f"Found {len(relations)} related concept(s) for CUI {cui}"
        }
        return json.dumps(response, ensure_ascii=False)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover
        raise NotImplementedError("Async operation is not supported.")


# ------------------------------------------------------------------------- #
# Multi-agent Orchestrator
# ------------------------------------------------------------------------- #


def _read_openai_key() -> Optional[str]:
    """Read OpenAI API key from .openai_api_key file or environment variable."""
    # Try reading from file first
    try:
        key = DEFAULT_OPENAI_KEY_PATH.read_text(encoding="utf-8").strip()
        if key:
            return key
    except FileNotFoundError:
        LOGGER.debug("OpenAI API key file not found at %s", DEFAULT_OPENAI_KEY_PATH)
    except Exception as e:
        LOGGER.warning("Error reading OpenAI API key file: %s", e)
    
    # Fall back to environment variable
    return os.environ.get("OPENAI_API_KEY")


def _default_llm(model: str = "gpt-4o-mini", temperature: float = 0.0) -> ChatOpenAI:
    """Factory for a deterministic ChatOpenAI instance."""
    api_key = _read_openai_key()
    return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)


def _extract_json_blob(text: str) -> Optional[Dict[str, Any]]:
    """Try to recover the outermost JSON object from a free-form string."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


class LangChainConceptGenerator:
    """High-level interface for orchestrating UMLS concept discovery."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        llm: Optional[ChatOpenAI] = None,
        llm_model: str = "gpt-4o-mini",
        llm_temperature: float = 0.0,
        verbose: bool = False,
    ) -> None:
        self.client = UMLSClient(api_key=api_key)
        self.llm = llm or _default_llm(model=llm_model, temperature=llm_temperature)
        self.verbose = verbose

        tools = [
            SearchUMLSTool(client=self.client),
            FetchConceptDetailsTool(client=self.client),
            FetchRelatedConceptsTool(client=self.client),
        ]

        planner = load_chat_planner(self.llm)
        executor = load_agent_executor(self.llm, tools, verbose=verbose)
        self.controller = PlanAndExecute(
            planner=planner,
            executor=executor,
            verbose=verbose,
        )

    def generate(
        self,
        disease_name: str,
        *,
        max_concepts: int = 5,
        relation_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run the multi-agent workflow for a single disease.
        
        Args:
            disease_name: Disease name (underscores will be replaced with spaces for search)
            max_concepts: Maximum number of candidate concepts to retrieve
            relation_types: Optional list of relation types to filter
        """
        # Replace underscores with spaces for better UMLS search
        search_name = disease_name.replace("_", " ")
        instructions = (
            "You are a biomedical knowledge mining agent. Your task is to search UMLS and extract concept information.\n"
            "\n"
            "IMPORTANT: DO NOT write or execute Python code. Work directly with the JSON responses from tools.\n"
            "\n"
            "STEP 1: Search for the disease.\n"
            "- Use search_umls_concepts tool with the disease name.\n"
            "- The tool returns JSON: {\"count\": N, \"results\": [...], \"message\": \"...\"}\n"
            "- Read the 'count' field. If count is 0, STOP and return empty candidate_concepts.\n"
            "- If count > 0, read the 'results' array. Each item has 'ui' (the CUI), 'name', 'score'.\n"
            "- Extract CUIs from the results array by reading the 'ui' field of each result.\n"
            "- DO NOT use Python list indexing like results[0]. Read the JSON structure directly.\n"
            "\n"
            f"STEP 2: Get details for up to {max_concepts} CUIs.\n"
            "- For each selected CUI, use fetch_umls_concept_details tool.\n"
            "- Extract: 'name', 'semanticTypes' (array), 'definition', source info from 'atoms'.\n"
            "\n"
            "STEP 3: Get related concepts for each CUI.\n"
            "- Use fetch_umls_related_concepts tool for each CUI.\n"
            "- Tool returns: {\"count\": N, \"relations\": [...], \"message\": \"...\"}\n"
            "- Read 'count' first. If count is 0, use empty array [] for that CUI.\n"
            "- If count > 0, read 'relations' array. Each has 'relatedId', 'relatedIdName', 'relationLabel'.\n"
            "\n"
            "STEP 4: Build the output JSON:\n"
            "{\n"
            '  "disease": "disease_name",\n'
            '  "candidate_concepts": [\n'
            "    {\n"
            '      "cui": "C1234567",\n'
            '      "name": "Concept Name",\n'
            '      "score": 100.0,\n'
            '      "semantic_types": ["Disease or Syndrome"],\n'
            '      "definition": "Definition or null",\n'
            '      "sources": ["SNOMEDCT_US"]\n'
            "    }\n"
            "  ],\n"
            '  "related_concepts": {\n'
            '    "C1234567": [\n'
            "      {\n"
            '        "related_cui": "C0987654",\n'
            '        "name": "Related Name",\n'
            '        "relation_label": "may_cause",\n'
            '        "relation": "RO",\n'
            '        "additional_information": null\n'
            "      }\n"
            "    ]\n"
            "  }\n"
            "}\n"
            "\n"
            "CRITICAL SAFETY RULES:\n"
            "1. NEVER write or execute Python code. Work with JSON text directly.\n"
            "2. ALWAYS check 'count' field before reading any array from tool responses.\n"
            "3. If count is 0, the array is empty - use [] in your output.\n"
            "4. Read JSON fields directly - don't parse with Python code.\n"
            "5. Return ONLY the JSON object - no markdown, no code, no explanations."
        )

        prompt = (
            f"Disease: {search_name}\n"
            f"Relation types constraint: {relation_types or 'None'}\n"
            f"{instructions}\n"
            "Return ONLY the JSON object — no extra narration."
        )

        if self.verbose:
            LOGGER.info("Starting concept generation workflow for '%s'", disease_name)
            LOGGER.debug("Prompt: %s", prompt[:500] + "..." if len(prompt) > 500 else prompt)

        try:
            raw_output = self.controller.invoke({"input": prompt})
            if self.verbose:
                LOGGER.debug("Raw agent output type: %s", type(raw_output))
                if isinstance(raw_output, dict):
                    LOGGER.debug("Raw agent output keys: %s", list(raw_output.keys()))
        except (IndexError, KeyError) as exc:
            # Handle index errors that might occur during agent execution
            LOGGER.warning(
                "Agent execution encountered an index/key error for '%s': %s. "
                "This may indicate empty search results or the agent tried to access "
                "list elements without checking if the list is empty. Trying direct API fallback.",
                disease_name, exc
            )
            if self.verbose:
                import traceback
                LOGGER.debug("Full traceback:\n%s", traceback.format_exc())
            # Try direct API fallback instead of returning error
            return self._direct_api_fallback(disease_name, max_concepts=max_concepts, relation_types=relation_types)
        except Exception as exc:
            # Catch any other exceptions during agent execution
            LOGGER.warning("Agent execution failed for '%s': %s. Trying direct API fallback.", disease_name, exc)
            if self.verbose:
                import traceback
                LOGGER.debug("Full traceback:\n%s", traceback.format_exc())
            # Try direct API fallback for any agent failure
            return self._direct_api_fallback(disease_name, max_concepts=max_concepts, relation_types=relation_types)

        if isinstance(raw_output, dict):
            # Depending on LangChain version this may already be a dict.
            candidate = raw_output.get("output", raw_output)
        else:
            candidate = raw_output

        parsed = _extract_json_blob(candidate if isinstance(candidate, str) else json.dumps(candidate))

        if not parsed:
            LOGGER.warning("Unable to parse agent output into JSON for '%s'. Trying direct API fallback.", disease_name)
            return self._direct_api_fallback(disease_name, max_concepts=max_concepts, relation_types=relation_types)

        parsed.setdefault("disease", disease_name)
        parsed["candidate_concepts"] = parsed.get("candidate_concepts", [])[:max_concepts]
        # Ensure related_concepts is a dict if not present
        parsed.setdefault("related_concepts", {})
        
        # Validate that we got some results
        if not parsed.get("candidate_concepts") and not parsed.get("error"):
            # If no concepts found and no error, try direct API fallback
            LOGGER.warning("Agent returned empty results for '%s', trying direct API fallback", disease_name)
            return self._direct_api_fallback(disease_name, max_concepts=max_concepts, relation_types=relation_types)
        
        return parsed
    
    def _direct_api_fallback(
        self,
        disease_name: str,
        *,
        max_concepts: int = 5,
        relation_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Fallback method that directly queries UMLS API without using agents."""
        LOGGER.info("Using direct API fallback for '%s'", disease_name)
        result = {
            "disease": disease_name,
            "candidate_concepts": [],
            "related_concepts": {},
        }
        
        try:
            # Direct search (replace underscores with spaces for better search)
            search_name = disease_name.replace("_", " ")
            search_results = self.client.search_concepts(search_name, page_size=max_concepts)
            
            if not search_results:
                LOGGER.warning("No search results found for '%s'", disease_name)
                return result
            
            # Process each result
            for item in search_results[:max_concepts]:
                cui = item.get("ui")
                if not cui:
                    continue
                
                # Get concept details
                try:
                    details = self.client.fetch_concept_details(cui)
                    # Handle case where details might be a string or not a dict
                    if not isinstance(details, dict):
                        LOGGER.debug("Details for CUI %s is not a dict: %s", cui, type(details))
                        details = {}
                    
                    concept_info = {
                        "cui": cui,
                        "name": details.get("name") or item.get("name", ""),
                        "score": float(item.get("score", 0)) if item.get("score") else None,
                        "semantic_types": [
                            st.get("name", "") 
                            for st in details.get("semanticTypes", [])
                            if isinstance(st, dict)
                        ],
                        "definition": details.get("definition") or None,
                        "sources": list(set(
                            atom.get("rootSource", "")
                            for atom in details.get("atoms", [])
                            if isinstance(atom, dict) and atom.get("rootSource")
                        )),
                    }
                    result["candidate_concepts"].append(concept_info)
                    
                    # Get related concepts
                    try:
                        relations = self.client.fetch_related_concepts(cui, relation_types=relation_types)
                        related_list = []
                        for rel in relations:
                            related_list.append({
                                "related_cui": rel.get("relatedId", ""),
                                "name": rel.get("relatedIdName"),
                                "relation_label": rel.get("relationLabel"),
                                "relation": rel.get("relation"),
                                "additional_information": None,
                            })
                        if related_list:
                            result["related_concepts"][cui] = related_list
                    except Exception as rel_exc:
                        LOGGER.debug("Failed to fetch relations for CUI %s: %s", cui, rel_exc)
                        
                except Exception as detail_exc:
                    LOGGER.debug("Failed to fetch details for CUI %s: %s", cui, detail_exc)
                    # Still add basic info if details fail
                    result["candidate_concepts"].append({
                        "cui": cui,
                        "name": item.get("name", ""),
                        "score": float(item.get("score", 0)) if item.get("score") else None,
                        "semantic_types": [],
                        "definition": None,
                        "sources": [],
                    })
                    
        except Exception as exc:
            LOGGER.exception("Direct API fallback failed for '%s'", disease_name)
            result["error"] = f"Direct API fallback failed: {str(exc)}"
        
        return result

    def generate_batch(
        self,
        diseases: Iterable[str],
        *,
        max_concepts: int = 5,
        relation_types: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Run the workflow for many diseases, collecting results."""
        results: Dict[str, Dict[str, Any]] = {}
        for disease in diseases:
            disease = disease.strip()
            if not disease:
                continue
            try:
                results[disease] = self.generate(
                    disease,
                    max_concepts=max_concepts,
                    relation_types=relation_types,
                )
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.exception("Failed to generate concepts for %s", disease)
                results[disease] = {"error": str(exc)}
        return results


# ------------------------------------------------------------------------- #
# Legacy helper functions preserved for compatibility
# ------------------------------------------------------------------------- #


def load_diseases(filepath: str) -> List[str]:
    """Load disease names from a file, excluding the 'No Finding' token.
    
    Also handles underscore-separated names (e.g., 'Pleural_thickening' -> 'Pleural thickening').
    """
    diseases = []
    with open(filepath, "r", encoding="utf-8") as handle:
        for line in handle:
            disease = line.strip()
            if disease and disease.lower() != "no finding":
                # Replace underscores with spaces for better UMLS search
                # But keep original for filename purposes
                diseases.append(disease)
    return diseases


def extract_cui_from_url(url_or_cui: str) -> Optional[str]:
    """Extract CUI from URL or return the CUI if it's already a CUI."""
    if not url_or_cui:
        return None
    
    # If it's already a CUI (starts with C and followed by digits)
    if re.match(r'^C\d+$', url_or_cui):
        return url_or_cui
    
    # Try to extract CUI from URL patterns
    # Pattern: /CUI/C1234567 or /content/2025AB/CUI/C1234567
    cui_match = re.search(r'/CUI/(C\d+)', url_or_cui)
    if cui_match:
        return cui_match.group(1)
    
    # Pattern: /AUI/A12345678 (Atomic Unique Identifier)
    aui_match = re.search(r'/AUI/(A\d+)', url_or_cui)
    if aui_match:
        return aui_match.group(1)
    
    return None


def generate_concept_set_from_related_concepts(
    disease_data: Dict[str, Any]
) -> List[str]:
    """Generate a text list of concept names from related concepts.
    
    Extracts unique concept names from all related concepts across all candidate concepts.
    Returns a deduplicated, sorted list of concept names.
    """
    concept_names: Set[str] = set()
    
    # Add candidate concept names
    for candidate in disease_data.get("candidate_concepts", []):
        name = candidate.get("name")
        if name:
            concept_names.add(name.strip())
    
    # Add related concept names
    related_concepts = disease_data.get("related_concepts", {})
    for cui, relations in related_concepts.items():
        if not isinstance(relations, list):
            continue
        for relation in relations:
            name = relation.get("name")
            if name:
                # Clean up the name (remove extra whitespace, normalize)
                cleaned_name = name.strip()
                if cleaned_name:
                    concept_names.add(cleaned_name)
    
    # Return sorted list for consistency
    return sorted(list(concept_names))


def save_concepts_to_json(
    concepts_dict: Dict[str, Any], 
    out_path: Optional[str] = None,
    *,
    use_timestamp_naming: bool = True,
    output_dir: Optional[str] = None,
    timestamp_dir: Optional[str] = None
) -> str:
    """Persist the concept dictionary to disk.
    
    Args:
        concepts_dict: Dictionary containing disease concepts
        out_path: Optional explicit output path (ignored if use_timestamp_naming is True)
        use_timestamp_naming: If True, use timestamp directory structure
        output_dir: Base directory (default: data/umls_concepts/)
        timestamp_dir: Optional pre-generated timestamp directory name
    
    Returns:
        Path to the timestamp directory or saved file
    """
    if use_timestamp_naming:
        # Determine base output directory
        if output_dir is None:
            base_dir = Path(__file__).resolve().parent / "data" / "umls_concepts"
        else:
            base_dir = Path(output_dir)
        
        # Generate or use provided timestamp directory
        if timestamp_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            timestamp = timestamp_dir
        
        # Create timestamped directory
        timestamp_dir_path = base_dir / timestamp
        timestamp_dir_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        for disease_name, disease_data in concepts_dict.items():
            # Sanitize disease name for filename (keep underscores, remove special chars)
            safe_disease_name = re.sub(r'[^\w\s_-]', '', disease_name).strip()
            safe_disease_name = re.sub(r'[-\s]+', '_', safe_disease_name)
            filename = f"{safe_disease_name}.json"
            file_path = timestamp_dir_path / filename
            
            # Save individual disease file
            with open(file_path, "w", encoding="utf-8") as handle:
                json.dump({disease_name: disease_data}, handle, indent=2, ensure_ascii=False)
            saved_files.append(str(file_path))
        
        # Return timestamp directory path
        return str(timestamp_dir_path)
    else:
        # Use explicit path
        if out_path is None:
            out_path = "data/umls_concepts.json"
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(concepts_dict, handle, indent=2, ensure_ascii=False)
        return out_path


def save_concept_set_to_txt(
    disease_data: Dict[str, Any],
    output_dir: Optional[str] = None,
    timestamp_dir: Optional[str] = None
) -> str:
    """Save concept set as a text file (one concept per line).
    
    Args:
        disease_data: Dictionary containing disease concepts (single disease)
        output_dir: Base directory (default: data/umls_concepts/)
        timestamp_dir: Optional pre-generated timestamp directory name
    
    Returns:
        Path to the saved file
    """
    # Determine base output directory
    if output_dir is None:
        base_dir = Path(__file__).resolve().parent / "data" / "umls_concepts"
    else:
        base_dir = Path(output_dir)
    
    # Generate or use provided timestamp directory
    if timestamp_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        timestamp = timestamp_dir
    
    # Create timestamped directory
    timestamp_dir_path = base_dir / timestamp
    timestamp_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Generate concept set
    concept_set = generate_concept_set_from_related_concepts(disease_data)
    
    # Generate filename (simple disease name, no timestamp)
    disease_name = disease_data.get("disease", "unknown")
    safe_disease_name = re.sub(r'[^\w\s_-]', '', disease_name).strip()
    safe_disease_name = re.sub(r'[-\s]+', '_', safe_disease_name)
    filename = f"{safe_disease_name}.txt"
    file_path = timestamp_dir_path / filename
    
    # Save text file
    with open(file_path, "w", encoding="utf-8") as handle:
        for concept in concept_set:
            handle.write(f"{concept}\n")
    
    return str(file_path)


def generate_umls_concepts_for_diseases(
    diseases: Iterable[str],
    *,
    api_key: Optional[str] = None,
    max_concepts: int = 5,
    relation_types: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Convenience wrapper to run the LangChain multi-agent system."""
    generator = LangChainConceptGenerator(api_key=api_key, verbose=verbose)
    return generator.generate_batch(
        diseases,
        max_concepts=max_concepts,
        relation_types=relation_types,
    )


# ------------------------------------------------------------------------- #
# CLI entry point
# ------------------------------------------------------------------------- #


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate UMLS concepts via a LangChain multi-agent workflow."
    )
    parser.add_argument(
        "--disease",
        type=str,
        help="Single disease name to process.",
    )
    parser.add_argument(
        "--disease-file",
        type=str,
        help="Path to a newline-delimited list of diseases.",
    )
    parser.add_argument(
        "--max-concepts",
        type=int,
        default=10,
        help="Maximum number of candidate concepts per disease.",
    )
    parser.add_argument(
        "--relation-types",
        type=str,
        nargs="*",
        help="Optional list of relation types to include (space separated).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging agent behaviour.",
    )
    parser.add_argument(
        "--from-json",
        type=str,
        help="Generate concept sets from existing JSON file instead of running API queries.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    # Handle generating concept sets from existing JSON
    if args.from_json:
        with open(args.from_json, "r", encoding="utf-8") as handle:
            results = json.load(handle)
        
        # Generate timestamp once for all files in this batch
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        LOGGER.info("Generating concept sets from %s", args.from_json)
        for disease_name, disease_data in results.items():
            if "error" not in disease_data:
                concept_set_path = save_concept_set_to_txt(
                    disease_data,
                    output_dir="data/umls_concepts",
                    timestamp_dir=timestamp
                )
                concept_set = generate_concept_set_from_related_concepts(disease_data)
                LOGGER.info(
                    "Generated concept set for '%s': %d concepts saved to %s",
                    disease_name,
                    len(concept_set),
                    concept_set_path
                )
        return
    
    # Normal workflow: generate concepts from API
    if not args.disease and not args.disease_file:
        raise ValueError("Provide either --disease or --disease-file.")

    diseases: List[str] = []
    if args.disease:
        diseases.append(args.disease)
    if args.disease_file:
        diseases.extend(load_diseases(args.disease_file))

    generator = LangChainConceptGenerator(verbose=args.verbose)
    results = generator.generate_batch(
        diseases,
        max_concepts=args.max_concepts,
        relation_types=args.relation_types,
    )
    
    # Generate timestamp once for all files in this batch
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON files with timestamp directory structure
    saved_path = save_concepts_to_json(
        results, 
        use_timestamp_naming=True,
        output_dir="data/umls_concepts",
        timestamp_dir=timestamp
    )
    LOGGER.info(
        "Saved UMLS concept generations for %d diseases to %s",
        len(results),
        saved_path,
    )
    
    # Generate and save concept sets (text files) from related concepts
    for disease_name, disease_data in results.items():
        if "error" not in disease_data:
            concept_set_path = save_concept_set_to_txt(
                disease_data,
                output_dir="data/umls_concepts",
                timestamp_dir=timestamp
            )
            concept_set = generate_concept_set_from_related_concepts(disease_data)
            LOGGER.info(
                "Generated concept set for '%s': %d concepts saved to %s",
                disease_name,
                len(concept_set),
                concept_set_path
            )


if __name__ == "__main__":  # pragma: no cover
    main()
