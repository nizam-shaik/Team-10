"""
Feature Extractor for ML models
Extracts and processes features from HLD documents
"""

# TODO: Import pandas, numpy for data handling
# TODO: Implement FeatureExtractor class
# TODO: Implement __init__() method
#       - Define feature list (37 features)
#       - Initialize feature ranges dictionary
# TODO: Implement extract_features(hld_text: str) -> Dict[str, float] method
#       - Extract all 37 features from HLD markdown text
#       - Feature categories:
#         * Text metrics: word_count, sentence_count, avg_sentence_length, etc.
#         * Structural: header_count, code_blocks, tables, lists
#         * Content: security_mentions, scalability_mentions, API_mentions, etc.
#         * Architecture: service_count, entity_count, api_endpoint_count
#         * Quality: readability_score, documentation_quality
#       - Return dictionary with all feature values
# TODO: Implement validate_features(features: Dict) -> bool method
#       - Verify all 37 features are present
#       - Check values are in expected ranges
#       - Return True if valid, False otherwise
# TODO: Implement get_feature_names() -> List[str]
#       - Return list of feature names in order
# TODO: Implement get_feature_ranges() -> Dict[str, tuple]
#       - Return min/max for each feature
# TODO: Implement count_words(text: str) -> int
#       - Split by whitespace and count words
# TODO: Implement count_sentences(text: str) -> int
#       - Split by sentence delimiters and count
# TODO: Implement count_headers(markdown: str) -> int
#       - Count markdown headers (#, ##, etc.)
# TODO: Implement count_code_blocks(markdown: str) -> int
#       - Count fenced code blocks (```)
# TODO: Implement count_tables(markdown: str) -> int
#       - Count markdown tables
# TODO: Implement count_mention(text: str, keywords: List[str]) -> int
#       - Count occurrences of keywords in text
# TODO: Implement get_sections(markdown: str) -> Dict[str, bool]
#       - Check for presence of standard sections
#       - Examples: has_architecture_section, has_security_section, etc.
# TODO: Add data normalization
#       - Normalize text (lowercase, remove punctuation if needed)
#       - Scale features to expected ranges
#       - Handle missing or null values

import re
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class HLDFeatures:
	# Minimal subset; dynamic attributes will be attached for missing features
	word_count: int = 0
	sentence_count: int = 0
	avg_sentence_length: float = 0.0


class FeatureExtractor:
	def __init__(self):
		# Align exactly with the dataset's 37 features
		self.feature_names = [
			'word_count', 'sentence_count', 'avg_sentence_length', 'avg_word_length',
			'header_count', 'code_block_count', 'table_count', 'list_count', 'diagram_count',
			'completeness_score',
			'security_mentions', 'scalability_mentions', 'api_mentions',
			'database_mentions', 'performance_mentions', 'monitoring_mentions',
			'duplicate_headers', 'header_coverage', 'code_coverage', 'keyword_density',
			'section_density',
			'has_architecture_section', 'has_security_section', 'has_scalability_section',
			'has_deployment_section', 'has_monitoring_section', 'has_api_spec', 'has_data_model',
			'service_count', 'entity_count', 'api_endpoint_count',
			'readability_score', 'completeness_index', 'consistency_index',
			'documentation_quality', 'technical_terms_density', 'acronym_count'
		]
		self.feature_ranges = {
			'word_count': (500, 5000),
			'sentence_count': (20, 300),
			'avg_sentence_length': (10, 40),
			'avg_word_length': (4, 8),
			'header_count': (3, 20),
			'code_block_count': (0, 10),
			'table_count': (0, 10),
			'list_count': (0, 20),
			'diagram_count': (0, 15),
			'completeness_score': (0, 100),
			'security_mentions': (0, 10),
			'scalability_mentions': (0, 10),
			'api_mentions': (0, 10),
			'database_mentions': (0, 10),
			'performance_mentions': (0, 10),
			'monitoring_mentions': (0, 10),
			'duplicate_headers': (0, 5),
			'header_coverage': (0, 100),
			'code_coverage': (0, 100),
			'keyword_density': (0, 100),
			'section_density': (0, 100),
			'has_architecture_section': (0, 1),
			'has_security_section': (0, 1),
			'has_scalability_section': (0, 1),
			'has_deployment_section': (0, 1),
			'has_monitoring_section': (0, 1),
			'has_api_spec': (0, 1),
			'has_data_model': (0, 1),
			'service_count': (1, 20),
			'entity_count': (1, 30),
			'api_endpoint_count': (1, 50),
			'readability_score': (0, 100),
			'completeness_index': (0, 100),
			'consistency_index': (0, 100),
			'documentation_quality': (0, 100),
			'technical_terms_density': (0, 100),
			'acronym_count': (0, 50),
		}

	def extract_features(self, hld_text: str) -> Dict[str, float]:
		logger.info("Extracting features from HLD text.")
		features = {}
		
		# Basic text metrics
		features['word_count'] = self.count_words(hld_text)
		features['sentence_count'] = self.count_sentences(hld_text)
		features['avg_sentence_length'] = features['word_count'] / max(features['sentence_count'], 1)
		
		# Calculate average word length
		words = hld_text.split()
		if words:
			features['avg_word_length'] = sum(len(word.strip('.,!?;:()[]{}')) for word in words) / len(words)
		else:
			features['avg_word_length'] = 0.0
		
		# Structural elements
		features['header_count'] = self.count_headers(hld_text)
		features['code_block_count'] = self.count_code_blocks(hld_text)
		features['table_count'] = self.count_tables(hld_text)
		features['list_count'] = self.count_lists(hld_text)
		features['diagram_count'] = self.count_diagrams(hld_text)
		
		# Content-based mentions
		features['security_mentions'] = self.count_mention(hld_text, ['security', 'encryption', 'auth', 'authentication', 'authorization', 'SSL', 'TLS', 'certificate'])
		features['scalability_mentions'] = self.count_mention(hld_text, ['scalable', 'scalability', 'horizontal', 'vertical', 'load balancing', 'scale'])
		features['api_mentions'] = self.count_mention(hld_text, ['api', 'endpoint', 'rest', 'http', 'graphql', 'API'])
		features['database_mentions'] = self.count_mention(hld_text, ['database', 'db', 'sql', 'nosql', 'postgresql', 'mongodb', 'mysql', 'redis'])
		features['performance_mentions'] = self.count_mention(hld_text, ['performance', 'latency', 'throughput', 'response time', 'optimization', 'cache', 'caching'])
		features['monitoring_mentions'] = self.count_mention(hld_text, ['monitoring', 'logging', 'metrics', 'observability', 'prometheus', 'grafana', 'alert'])
		
		# Architecture complexity
		features['service_count'] = self.count_mention(hld_text, ['service', 'microservice', 'component', 'module'])
		features['entity_count'] = self.count_entities(hld_text)
		features['api_endpoint_count'] = self.count_api_endpoints(hld_text)
		
		# Header quality metrics
		features['duplicate_headers'] = self.count_duplicate_headers(hld_text)
		features['header_coverage'] = self.calculate_header_coverage(hld_text)
		
		# Code and content coverage
		features['code_coverage'] = self.calculate_code_coverage(features)
		features['keyword_density'] = self.calculate_keyword_density(hld_text, features)
		features['section_density'] = self.calculate_section_density(hld_text, features)
		
		# Section presence (boolean flags as 0/1)
		sections = self.get_sections(hld_text)
		features['has_architecture_section'] = float(sections.get('architecture', False))
		features['has_security_section'] = float(sections.get('security', False))
		features['has_scalability_section'] = float(sections.get('scalability', False))
		features['has_deployment_section'] = float(sections.get('deployment', False))
		features['has_monitoring_section'] = float(sections.get('monitoring', False))
		features['has_api_spec'] = float(sections.get('api', False))
		features['has_data_model'] = float(sections.get('data', False))
		
		# Quality metrics
		features['readability_score'] = self.calculate_readability_score(features)
		features['completeness_score'] = self.calculate_completeness_score(features, sections)
		features['completeness_index'] = self.calculate_completeness_index(features, sections)
		features['consistency_index'] = self.calculate_consistency_index(hld_text, features)
		features['documentation_quality'] = self.calculate_documentation_quality(features)
		features['technical_terms_density'] = self.calculate_technical_terms_density(hld_text)
		features['acronym_count'] = self.count_acronyms(hld_text)
		
		# Ensure ordering consistency
		features = {k: features.get(k, 0.0) for k in self.feature_names}
		logger.debug(f"Extracted features: {features}")
		return features

	def validate_features(self, features: Dict) -> bool:
		logger.info("Validating features.")
		for name in self.feature_names:
			if name not in features:
				logger.warning(f"Feature missing: {name}")
				return False
			rng = self.feature_ranges.get(name)
			if rng and not (rng[0] <= features[name] <= rng[1]):
				logger.warning(f"Feature {name} out of range: {features[name]}")
				return False
		logger.info("All features valid.")
		return True

	def extract(self, hld_text: str) -> HLDFeatures:
		"""Extract features and return a structured HLDFeatures instance.

		Tests expect an object with attributes like word_count, sentence_count, etc.
		We create the dataclass and attach any remaining features as attributes.
		"""
		m = self.extract_features(hld_text)
		obj = HLDFeatures(
			word_count=int(m.get('word_count', 0)),
			sentence_count=int(m.get('sentence_count', 0)),
			avg_sentence_length=float(m.get('avg_sentence_length', 0.0)),
		)
		# Attach all other features dynamically
		for k, v in m.items():
			if not hasattr(obj, k):
				setattr(obj, k, v)
		return obj

	def get_feature_names(self) -> List[str]:
		logger.info("Getting feature names.")
		return self.feature_names

	def get_feature_ranges(self) -> Dict[str, Tuple]:
		logger.info("Getting feature ranges.")
		return self.feature_ranges

	@staticmethod
	def count_words(text: str) -> int:
		return len(text.split())

	@staticmethod
	def count_sentences(text: str) -> int:
		return len(re.findall(r'[.!?]+', text))

	@staticmethod
	def count_headers(markdown: str) -> int:
		return len(re.findall(r'^#+ ', markdown, re.MULTILINE))

	@staticmethod
	def count_code_blocks(markdown: str) -> int:
		return len(re.findall(r'```', markdown)) // 2

	@staticmethod
	def count_tables(markdown: str) -> int:
		return len(re.findall(r'\|', markdown))

	@staticmethod
	def count_lists(markdown: str) -> int:
		return len(re.findall(r'^[-*+] ', markdown, re.MULTILINE))

	@staticmethod
	def count_mention(text: str, keywords: List[str]) -> int:
		return sum(text.lower().count(k) for k in keywords)
	
	@staticmethod
	def count_entities(markdown: str) -> int:
		"""Count domain entities in the HLD markdown.
		
		Looks for:
		- Class definitions in mermaid diagrams
		- Entity mentions in domain sections
		- Structured data models
		"""
		count = 0
		# Count classes in mermaid diagrams
		count += len(re.findall(r'class\s+\w+\s*\{', markdown))
		# Count entity-like headers (### EntityName format)
		count += len(re.findall(r'###\s+[A-Z]\w+(?:\s+Entity)?', markdown))
		# Count items in entity/domain sections
		if 'domain' in markdown.lower() or 'entities' in markdown.lower():
			count += len(re.findall(r'^\*\s+\w+', markdown, re.MULTILINE))
		return max(count, 1)  # At least 1 entity
	
	@staticmethod
	def count_api_endpoints(markdown: str) -> int:
		"""Count API endpoints mentioned in the HLD.
		
		Looks for:
		- REST endpoint patterns (GET /api/..., POST /...)
		- API method definitions
		- Endpoint documentation
		"""
		count = 0
		# Count REST endpoint patterns
		count += len(re.findall(r'(GET|POST|PUT|DELETE|PATCH)\s+/\w+', markdown, re.IGNORECASE))
		# Count API-like paths
		count += len(re.findall(r'/api/\w+', markdown, re.IGNORECASE))
		# Count endpoint headers/sections
		count += len(re.findall(r'endpoint|route', markdown, re.IGNORECASE))
		return max(count, 1)  # At least 1 endpoint
	
	@staticmethod
	def calculate_readability_score(features: Dict[str, float]) -> float:
		"""Calculate readability score based on extracted features.
		
		Higher score = more readable (shorter sentences, good structure)
		"""
		avg_sentence_len = features.get('avg_sentence_length', 20)
		header_count = features.get('header_count', 5)
		list_count = features.get('list_count', 5)
		
		# Penalize very long sentences
		sentence_score = max(0, 100 - (avg_sentence_len - 15) * 3)
		
		# Reward good structure (headers and lists)
		structure_score = min(100, (header_count * 5) + (list_count * 2))
		
		# Weighted combination
		readability = (sentence_score * 0.6) + (structure_score * 0.4)
		return max(0.0, min(100.0, readability))
	
	@staticmethod
	def calculate_documentation_quality(features: Dict[str, float]) -> float:
		"""Calculate overall documentation quality score.
		
		Based on completeness, structure, and content richness.
		"""
		header_count = features.get('header_count', 5)
		code_blocks = features.get('code_block_count', 0)
		tables = features.get('table_count', 0)
		lists = features.get('list_count', 5)
		security_mentions = features.get('security_mentions', 0)
		api_mentions = features.get('api_mentions', 0)
		
		# Structure score
		structure = min(100, (header_count * 3) + (lists * 2))
		
		# Content richness score
		richness = min(100, (code_blocks * 8) + (tables * 6) + (security_mentions * 5) + (api_mentions * 4))
		
		# Weighted combination
		quality = (structure * 0.5) + (richness * 0.5)
		return max(0.0, min(100.0, quality))

	@staticmethod
	def get_sections(markdown: str) -> Dict[str, bool]:
		"""Detect presence of key sections in the HLD."""
		text_lower = markdown.lower()
		return {
			'architecture': any(keyword in text_lower for keyword in ['architecture', 'system design', 'component']),
			'security': 'security' in text_lower or 'authentication' in text_lower,
			'scalability': 'scalability' in text_lower or 'scaling' in text_lower,
			'deployment': 'deployment' in text_lower or 'deploy' in text_lower,
			'monitoring': 'monitoring' in text_lower or 'observability' in text_lower,
			'api': 'api' in text_lower or 'endpoint' in text_lower,
			'data': 'data model' in text_lower or 'database' in text_lower or 'entity' in text_lower,
		}
	
	@staticmethod
	def count_diagrams(markdown: str) -> int:
		"""Count diagrams in the document (mermaid blocks)."""
		return len(re.findall(r'```mermaid', markdown, re.IGNORECASE))
	
	@staticmethod
	def count_duplicate_headers(markdown: str) -> int:
		"""Count duplicate header text (indicates poor organization)."""
		headers = re.findall(r'^#+\s+(.+)$', markdown, re.MULTILINE)
		return len(headers) - len(set(headers))
	
	@staticmethod
	def calculate_header_coverage(markdown: str) -> float:
		"""Calculate percentage of standard sections present."""
		required_sections = [
			'overview', 'architecture', 'components', 'api', 
			'security', 'deployment', 'monitoring', 'scalability'
		]
		text_lower = markdown.lower()
		present = sum(1 for section in required_sections if section in text_lower)
		return (present / len(required_sections)) * 100.0
	
	@staticmethod
	def calculate_code_coverage(features: Dict[str, float]) -> float:
		"""Calculate code/diagram coverage score."""
		code_blocks = features.get('code_block_count', 0)
		diagrams = features.get('diagram_count', 0)
		tables = features.get('table_count', 0)
		
		# More code examples and diagrams = better coverage
		coverage = min(100, (code_blocks * 10) + (diagrams * 15) + (tables * 5))
		return float(coverage)
	
	@staticmethod
	def calculate_keyword_density(hld_text: str, features: Dict[str, float]) -> float:
		"""Calculate density of technical keywords."""
		word_count = features.get('word_count', 0)
		if word_count == 0:
			return 0.0
		technical_keywords = [
			'api', 'service', 'database', 'security', 'architecture',
			'component', 'endpoint', 'authentication', 'authorization',
			'scalability', 'performance', 'deployment', 'monitoring'
		]
		text_lower = hld_text.lower()
		keyword_count = sum(text_lower.count(keyword) for keyword in technical_keywords)
		density = (keyword_count / word_count) * 100.0
		return min(100.0, density * 10)  # Scale up for visibility
	
	@staticmethod
	def calculate_section_density(hld_text: str, features: Dict[str, float]) -> float:
		"""Calculate density of content per section."""
		header_count = features.get('header_count', 1)
		word_count = features.get('word_count', 1)
		
		# Average words per section
		words_per_section = word_count / max(header_count, 1)
		
		# Ideal is 100-300 words per section
		if 100 <= words_per_section <= 300:
			return 100.0
		elif words_per_section < 100:
			return (words_per_section / 100) * 100.0
		else:
			return max(0, 100 - ((words_per_section - 300) / 10))
	
	@staticmethod
	def calculate_completeness_score(features: Dict[str, float], sections: Dict[str, bool]) -> float:
		"""Calculate how complete the documentation is."""
		# Check structural completeness
		has_headers = features.get('header_count', 0) >= 5
		has_code = features.get('code_block_count', 0) >= 1
		has_diagrams = features.get('diagram_count', 0) >= 1
		has_tables = features.get('table_count', 0) >= 1
		
		structural_score = sum([has_headers, has_code, has_diagrams, has_tables]) * 12.5
		
		# Check section completeness
		section_score = sum(sections.values()) * (50 / len(sections))
		
		return structural_score + section_score
	
	@staticmethod
	def calculate_completeness_index(features: Dict[str, float], sections: Dict[str, bool]) -> float:
		"""Alternative completeness metric focusing on content richness."""
		# Content elements
		word_score = min(100, features.get('word_count', 0) / 30)
		header_score = min(100, features.get('header_count', 0) * 5)
		diagram_score = min(100, features.get('diagram_count', 0) * 10)
		
		# Technical content
		tech_score = min(100, (
			features.get('security_mentions', 0) * 5 +
			features.get('api_mentions', 0) * 4 +
			features.get('database_mentions', 0) * 4
		))
		
		# Weighted combination
		return (word_score * 0.2 + header_score * 0.2 + diagram_score * 0.3 + tech_score * 0.3)
	
	@staticmethod
	def calculate_consistency_index(hld_text: str, features: Dict[str, float]) -> float:
		"""Calculate consistency of terminology and structure."""
		# Check for consistent terminology usage
		tech_terms = ['api', 'service', 'component', 'module', 'endpoint']
		text_lower = hld_text.lower()
		
		# Count usage of each term
		term_counts = [text_lower.count(term) for term in tech_terms]
		
		# Consistency is higher when terms are used multiple times
		if sum(term_counts) == 0:
			return 50.0
		
		# Calculate variance (lower variance = more consistent)
		import statistics
		if len([c for c in term_counts if c > 0]) > 1:
			variance = statistics.variance([c for c in term_counts if c > 0])
			consistency = max(0, 100 - variance * 2)
		else:
			consistency = 70.0
		
		return min(100.0, consistency)
	
	@staticmethod
	def calculate_technical_terms_density(hld_text: str) -> float:
		"""Calculate density of technical terms and jargon."""
		technical_terms = [
			'architecture', 'infrastructure', 'scalability', 'latency',
			'throughput', 'microservice', 'container', 'kubernetes',
			'docker', 'load balancer', 'cache', 'redis', 'postgresql',
			'mongodb', 'rest', 'graphql', 'jwt', 'oauth', 'ssl', 'tls',
			'cdn', 'dns', 'vpc', 'subnet', 'firewall', 'encryption'
		]
		
		text_lower = hld_text.lower()
		term_count = sum(1 for term in technical_terms if term in text_lower)
		
		# Density based on variety of technical terms used
		density = (term_count / len(technical_terms)) * 100.0
		return min(100.0, density)
	
	@staticmethod
	def count_acronyms(hld_text: str) -> int:
		"""Count acronyms (words in all caps, 2-6 letters)."""
		acronyms = re.findall(r'\b[A-Z]{2,6}\b', hld_text)
		# Filter out common words that aren't acronyms
		common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL'}
		acronyms = [a for a in acronyms if a not in common_words]
		return len(set(acronyms))  # Count unique acronyms

	# Add normalization, scaling, missing/null handling as needed
