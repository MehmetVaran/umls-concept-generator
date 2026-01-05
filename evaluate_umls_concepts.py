#!/usr/bin/env python3
"""Evaluation framework for UMLS-generated concept sets.

This module provides multiple evaluation metrics to assess the quality,
relevance, and diversity of UMLS concept sets generated for diseases.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Some metrics will be disabled.")

LOGGER = logging.getLogger(__name__)


class UMLSConceptEvaluator:
    """Evaluator for UMLS concept sets with multiple scoring metrics."""
    
    def __init__(self, use_embeddings: bool = True):
        """Initialize the evaluator.
        
        Args:
            use_embeddings: Whether to use sentence transformers for semantic similarity
        """
        self.use_embeddings = use_embeddings and SENTENCE_TRANSFORMERS_AVAILABLE
        if self.use_embeddings:
            try:
                self.model = SentenceTransformer('all-mpnet-base-v2')
                LOGGER.info("Loaded sentence transformer model for semantic similarity")
            except Exception as e:
                LOGGER.warning("Failed to load sentence transformer: %s", e)
                self.use_embeddings = False
        else:
            self.model = None
    
    def evaluate_concept_set(
        self,
        disease_name: str,
        disease_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate a concept set for a given disease.
        
        Returns a dictionary with various evaluation metrics.
        """
        metrics = {
            "disease": disease_name,
            "total_concepts": 0,
            "candidate_concepts_count": 0,
            "related_concepts_count": 0,
            "unique_concept_names": 0,
        }
        
        # Extract all concept names
        all_concepts = self._extract_all_concepts(disease_data)
        metrics["total_concepts"] = len(all_concepts)
        metrics["unique_concept_names"] = len(set(all_concepts))
        
        # Count candidate vs related concepts
        candidate_concepts = disease_data.get("candidate_concepts", [])
        metrics["candidate_concepts_count"] = len(candidate_concepts)
        
        related_count = sum(
            len(relations) 
            for relations in disease_data.get("related_concepts", {}).values()
        )
        metrics["related_concepts_count"] = related_count
        
        # Semantic type analysis
        semantic_types = self._analyze_semantic_types(disease_data)
        metrics["semantic_types"] = semantic_types
        
        # Relation type analysis
        relation_types = self._analyze_relation_types(disease_data)
        metrics["relation_types"] = relation_types
        
        # Language diversity
        language_diversity = self._analyze_language_diversity(all_concepts)
        metrics["language_diversity"] = language_diversity
        
        # Concept quality metrics
        quality_metrics = self._analyze_concept_quality(all_concepts, disease_name)
        metrics.update(quality_metrics)
        
        # Relevance scoring (if embeddings available)
        if self.use_embeddings:
            relevance_scores = self._calculate_relevance_scores(all_concepts, disease_name)
            metrics["relevance_scores"] = relevance_scores
        
        # Diversity scoring
        diversity_score = self._calculate_diversity_score(all_concepts)
        metrics["diversity_score"] = diversity_score
        
        # Overall score (weighted combination)
        overall_score = self._calculate_overall_score(metrics)
        metrics["overall_score"] = overall_score
        
        return metrics
    
    def _extract_all_concepts(self, disease_data: Dict[str, Any]) -> List[str]:
        """Extract all concept names from disease data."""
        concepts = []
        
        # Add candidate concepts
        for candidate in disease_data.get("candidate_concepts", []):
            name = candidate.get("name")
            if name:
                concepts.append(name)
        
        # Add related concepts
        for relations in disease_data.get("related_concepts", {}).values():
            if isinstance(relations, list):
                for relation in relations:
                    name = relation.get("name")
                    if name:
                        concepts.append(name)
        
        return concepts
    
    def _analyze_semantic_types(self, disease_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze semantic type distribution."""
        semantic_types = []
        for candidate in disease_data.get("candidate_concepts", []):
            stypes = candidate.get("semantic_types", [])
            if isinstance(stypes, list):
                semantic_types.extend(stypes)
        
        type_counts = Counter(semantic_types)
        return {
            "unique_types": len(type_counts),
            "type_distribution": dict(type_counts),
            "most_common": type_counts.most_common(5) if type_counts else []
        }
    
    def _analyze_relation_types(self, disease_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze relation type distribution."""
        relation_labels = []
        for relations in disease_data.get("related_concepts", {}).values():
            if isinstance(relations, list):
                for relation in relations:
                    label = relation.get("relation_label")
                    if label:
                        relation_labels.append(label)
        
        label_counts = Counter(relation_labels)
        return {
            "unique_relations": len(label_counts),
            "relation_distribution": dict(label_counts),
            "most_common": label_counts.most_common(10) if label_counts else []
        }
    
    def _analyze_language_diversity(self, concepts: List[str]) -> Dict[str, Any]:
        """Analyze language diversity in concept names."""
        # Simple heuristic: check for non-ASCII characters
        languages = {"english": 0, "other": 0}
        non_english_pattern = re.compile(r'[^\x00-\x7F]')
        
        for concept in concepts:
            if non_english_pattern.search(concept):
                languages["other"] += 1
            else:
                languages["english"] += 1
        
        return {
            "english_count": languages["english"],
            "other_languages_count": languages["other"],
            "english_ratio": languages["english"] / len(concepts) if concepts else 0
        }
    
    def _analyze_concept_quality(
        self, 
        concepts: List[str], 
        disease_name: str
    ) -> Dict[str, Any]:
        """Analyze concept quality metrics."""
        # Filter out noise
        noise_patterns = [
            r'^[A-Z\s]+$',  # All caps (often metadata)
            r'^[a-z\s]+$',  # All lowercase generic terms
            r'^\d+$',  # Pure numbers
            r'^[^\w\s]+$',  # Only special characters
        ]
        
        noise_count = 0
        quality_concepts = []
        
        for concept in concepts:
            is_noise = False
            for pattern in noise_patterns:
                if re.match(pattern, concept.strip()):
                    is_noise = True
                    break
            
            # Also check for very short or very long concepts
            if len(concept.strip()) < 3 or len(concept.strip()) > 100:
                is_noise = True
            
            if is_noise:
                noise_count += 1
            else:
                quality_concepts.append(concept)
        
        # Check for duplicates (case-insensitive)
        lower_concepts = [c.lower().strip() for c in quality_concepts]
        unique_lower = len(set(lower_concepts))
        duplicate_count = len(quality_concepts) - unique_lower
        
        return {
            "noise_count": noise_count,
            "quality_concepts_count": len(quality_concepts),
            "duplicate_count": duplicate_count,
            "unique_quality_concepts": unique_lower,
            "noise_ratio": noise_count / len(concepts) if concepts else 0,
            "duplicate_ratio": duplicate_count / len(quality_concepts) if quality_concepts else 0
        }
    
    def _calculate_relevance_scores(
        self, 
        concepts: List[str], 
        disease_name: str
    ) -> Dict[str, Any]:
        """Calculate semantic relevance scores using embeddings."""
        if not self.use_embeddings or not concepts:
            return {"average_relevance": 0.0, "min_relevance": 0.0, "max_relevance": 0.0}
        
        try:
            # Encode disease name and concepts
            disease_embedding = self.model.encode([disease_name])
            concept_embeddings = self.model.encode(concepts)
            
            # Calculate cosine similarities
            similarities = np.dot(concept_embeddings, disease_embedding.T).flatten()
            
            return {
                "average_relevance": float(np.mean(similarities)),
                "min_relevance": float(np.min(similarities)),
                "max_relevance": float(np.max(similarities)),
                "median_relevance": float(np.median(similarities)),
                "std_relevance": float(np.std(similarities))
            }
        except Exception as e:
            LOGGER.warning("Failed to calculate relevance scores: %s", e)
            return {"average_relevance": 0.0, "min_relevance": 0.0, "max_relevance": 0.0}
    
    def _calculate_diversity_score(self, concepts: List[str]) -> Dict[str, Any]:
        """Calculate diversity score based on concept uniqueness."""
        if not concepts:
            return {"diversity_score": 0.0, "unique_ratio": 0.0}
        
        unique_concepts = set(concepts)
        diversity_score = len(unique_concepts) / len(concepts)
        
        # Also check for similar concepts (simple word overlap)
        word_sets = [set(c.lower().split()) for c in unique_concepts]
        avg_word_overlap = 0.0
        if len(word_sets) > 1:
            overlaps = []
            for i in range(len(word_sets)):
                for j in range(i + 1, len(word_sets)):
                    if word_sets[i] and word_sets[j]:
                        overlap = len(word_sets[i] & word_sets[j]) / len(word_sets[i] | word_sets[j])
                        overlaps.append(overlap)
            if overlaps:
                avg_word_overlap = np.mean(overlaps) if SENTENCE_TRANSFORMERS_AVAILABLE else sum(overlaps) / len(overlaps)
        
        return {
            "diversity_score": diversity_score,
            "unique_ratio": diversity_score,
            "average_word_overlap": avg_word_overlap
        }
    
    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall weighted score."""
        scores = {}
        
        # Coverage score (0-1): based on number of concepts
        total_concepts = metrics.get("total_concepts", 0)
        coverage_score = min(total_concepts / 50.0, 1.0)  # Normalize to 50 concepts
        scores["coverage"] = coverage_score
        
        # Quality score (0-1): based on noise and duplicates
        noise_ratio = metrics.get("noise_ratio", 1.0)
        duplicate_ratio = metrics.get("duplicate_ratio", 1.0)
        quality_score = 1.0 - (noise_ratio * 0.5 + duplicate_ratio * 0.5)
        quality_score = max(0.0, quality_score)
        scores["quality"] = quality_score
        
        # Diversity score (0-1)
        diversity = metrics.get("diversity_score", {}).get("diversity_score", 0.0)
        scores["diversity"] = diversity
        
        # Relevance score (0-1): if available
        relevance_scores = metrics.get("relevance_scores", {})
        if relevance_scores and "average_relevance" in relevance_scores:
            # Normalize relevance (typically ranges from -1 to 1, we want 0 to 1)
            avg_rel = relevance_scores["average_relevance"]
            relevance_score = (avg_rel + 1) / 2.0  # Normalize to 0-1
            scores["relevance"] = relevance_score
        else:
            scores["relevance"] = 0.5  # Neutral if not available
        
        # Semantic type diversity (0-1)
        semantic_types = metrics.get("semantic_types", {})
        unique_types = semantic_types.get("unique_types", 0)
        type_diversity = min(unique_types / 10.0, 1.0)  # Normalize to 10 types
        scores["semantic_diversity"] = type_diversity
        
        # Adjust weights dynamically based on available data
        # If semantic types aren't available (e.g., loading from .txt files),
        # redistribute that weight proportionally to other metrics
        has_semantic_types = unique_types > 0
        
        if has_semantic_types:
            weights = {
                "coverage": 0.2,
                "quality": 0.3,
                "diversity": 0.2,
                "relevance": 0.2,
                "semantic_diversity": 0.1
            }
        else:
            # Redistribute semantic_diversity weight (0.1) proportionally
            weights = {
                "coverage": 0.222,      # 0.2 + 0.022
                "quality": 0.333,       # 0.3 + 0.033
                "diversity": 0.222,     # 0.2 + 0.022
                "relevance": 0.222,     # 0.2 + 0.022
                "semantic_diversity": 0.0
            }
        
        overall = sum(scores[key] * weights[key] for key in weights.keys())
        
        return {
            "component_scores": scores,
            "overall_score": overall,
            "weights": weights,
            "note": "Semantic type weights redistributed (not available in .txt format)" if not has_semantic_types else None
        }
    
    def generate_evaluation_report(
        self,
        disease_data: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """Generate a human-readable evaluation report."""
        disease_name = disease_data.get("disease", "Unknown")
        metrics = self.evaluate_concept_set(disease_name, disease_data)
        
        report_lines = [
            "=" * 80,
            f"UMLS Concept Set Evaluation Report: {disease_name}",
            "=" * 80,
            "",
            "SUMMARY METRICS",
            "-" * 80,
            f"Total Concepts: {metrics['total_concepts']}",
            f"Unique Concept Names: {metrics['unique_concept_names']}",
            f"Candidate Concepts: {metrics['candidate_concepts_count']}",
            f"Related Concepts: {metrics['related_concepts_count']}",
            "",
            "OVERALL SCORE",
            "-" * 80,
        ]
        
        overall = metrics.get("overall_score", {})
        if isinstance(overall, dict):
            report_lines.append(f"Overall Score: {overall.get('overall_score', 0):.3f} / 1.0")
            report_lines.append("")
            report_lines.append("Component Scores:")
            component_scores = overall.get("component_scores", {})
            for component, score in component_scores.items():
                report_lines.append(f"  - {component.capitalize()}: {score:.3f}")
        
        report_lines.extend([
            "",
            "QUALITY METRICS",
            "-" * 80,
            f"Noise Count: {metrics.get('noise_count', 0)}",
            f"Quality Concepts: {metrics.get('quality_concepts_count', 0)}",
            f"Duplicates: {metrics.get('duplicate_count', 0)}",
            f"Noise Ratio: {metrics.get('noise_ratio', 0):.2%}",
            f"Duplicate Ratio: {metrics.get('duplicate_ratio', 0):.2%}",
            "",
            "SEMANTIC TYPES",
            "-" * 80,
            f"Unique Semantic Types: {metrics.get('semantic_types', {}).get('unique_types', 0)}",
        ])
        
        most_common_types = metrics.get("semantic_types", {}).get("most_common", [])
        if most_common_types:
            report_lines.append("Most Common Types:")
            for stype, count in most_common_types:
                report_lines.append(f"  - {stype}: {count}")
        
        report_lines.extend([
            "",
            "RELATION TYPES",
            "-" * 80,
            f"Unique Relation Types: {metrics.get('relation_types', {}).get('unique_relations', 0)}",
        ])
        
        most_common_relations = metrics.get("relation_types", {}).get("most_common", [])
        if most_common_relations:
            report_lines.append("Most Common Relations:")
            for rel, count in most_common_relations[:5]:
                report_lines.append(f"  - {rel}: {count}")
        
        if metrics.get("relevance_scores"):
            report_lines.extend([
                "",
                "RELEVANCE SCORES (Semantic Similarity)",
                "-" * 80,
                f"Average Relevance: {metrics['relevance_scores'].get('average_relevance', 0):.3f}",
                f"Min Relevance: {metrics['relevance_scores'].get('min_relevance', 0):.3f}",
                f"Max Relevance: {metrics['relevance_scores'].get('max_relevance', 0):.3f}",
            ])
        
        report_lines.extend([
            "",
            "DIVERSITY",
            "-" * 80,
            f"Diversity Score: {metrics.get('diversity_score', {}).get('diversity_score', 0):.3f}",
            f"Unique Ratio: {metrics.get('diversity_score', {}).get('unique_ratio', 0):.3f}",
            "",
            "=" * 80,
        ])
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
            LOGGER.info("Evaluation report saved to %s", output_path)
        
        return report


def load_disease_data(data_path: str) -> Dict[str, Dict[str, Any]]:
    """Load disease data from .txt files.
    
    Args:
        data_path: Path to a single .txt file or directory containing .txt files.
                   Disease name is the filename (without extension).
                   Concepts are listed one per line in the file.
    
    Returns:
        Dictionary mapping disease names to their concept data.
    """
    p = Path(data_path)

    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {data_path}")

    # If a directory is provided, load all .txt files inside as separate diseases
    if p.is_dir():
        results: Dict[str, Dict[str, Any]] = {}
        txt_files = sorted(p.glob("*.txt"))
        
        if not txt_files:
            LOGGER.warning("No .txt files found in directory: %s", data_path)
            return results
        
        for txt in txt_files:
            disease_name = txt.stem
            lines = [l.strip() for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
            candidates = [{"name": name} for name in lines]
            results[disease_name] = {
                "disease": disease_name,
                "candidate_concepts": candidates,
                "related_concepts": {},
            }
            LOGGER.debug("Loaded %d concepts for disease: %s", len(candidates), disease_name)
        
        return results

    # If a single text file is provided, treat it as one disease (filename -> disease)
    if p.suffix.lower() == ".txt":
        disease_name = p.stem
        lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        candidates = [{"name": name} for name in lines]
        LOGGER.debug("Loaded %d concepts for disease: %s", len(candidates), disease_name)
        return {
            disease_name: {
                "disease": disease_name,
                "candidate_concepts": candidates,
                "related_concepts": {},
            }
        }

    raise ValueError(f"Expected a .txt file or directory containing .txt files. Got: {data_path}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate UMLS concept sets with multiple metrics."
    )
    parser.add_argument(
        "data_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to a .txt file or directory containing .txt files.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        dest="data_path_option",
        help="Alternative way to specify data path.",
    )
    parser.add_argument(
        "--disease",
        type=str,
        help="Specific disease to evaluate (if not provided, evaluates all).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for evaluation report (default: prints to stdout).",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Disable semantic similarity calculations (faster).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    
    args = parser.parse_args(argv)
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    # Determine which path to use
    data_path = args.data_path_option or args.data_path
    if not data_path:
        parser.error("Please provide a data path as either a positional argument or using --data-path")
    
    LOGGER.info("Loading data from: %s", data_path)
    
    try:
        # Load data
        data = load_disease_data(data_path)
    except (FileNotFoundError, ValueError) as e:
        LOGGER.error("Failed to load data: %s", e)
        return
    
    if not data:
        LOGGER.error("No data loaded from: %s", data_path)
        return
    
    LOGGER.info("Loaded %d disease(s)", len(data))
    
    # Initialize evaluator
    evaluator = UMLSConceptEvaluator(use_embeddings=not args.no_embeddings)
    
    # Evaluate
    if args.disease:
        if args.disease not in data:
            LOGGER.error("Disease '%s' not found in data", args.disease)
            return
        disease_data = data[args.disease]
        report = evaluator.generate_evaluation_report(disease_data, args.output)
        if not args.output:
            print(report)
    else:
        # Evaluate all diseases
        reports = []
        for disease_name, disease_data in data.items():
            LOGGER.info("Evaluating %s...", disease_name)
            report = evaluator.generate_evaluation_report(disease_data)
            reports.append(report)
        
        combined_report = "\n\n".join(reports)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(combined_report)
            LOGGER.info("Combined evaluation report saved to %s", args.output)
        else:
            print(combined_report)


if __name__ == "__main__":
    main()

