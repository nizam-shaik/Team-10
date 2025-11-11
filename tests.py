"""
Comprehensive Test Suite for DesignMind GenAI LangGraph System
Includes tests for workflow, state management, agents, ML models, and utilities
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# ============================================================================
# TEST 1: State Management and Models
# ============================================================================
class TestStateManagement:
    """Test 1: State management models and operations"""

    def test_hld_state_creation(self):
        """Test HLD state creation with valid data"""
        from state.models import HLDState

        state = HLDState(pdf_path="test.pdf", requirement_name="test")

        assert state.pdf_path == "test.pdf"
        assert state.requirement_name == "test"
        assert len(state.errors) == 0
        assert len(state.warnings) == 0
        assert isinstance(state.status, dict)


# ============================================================================
# TEST 2: Configuration Schema Validation
# ============================================================================
class TestConfigurationSchema:
    """Test 2: Configuration schema validation and defaults"""

    def test_config_schema_creation(self):
        """Test configuration schema with valid parameters"""
        from state.schema import ConfigSchema

        config = ConfigSchema(
            render_images=True,
            image_format="svg",
            renderer="kroki",
            theme="default"
        )

        assert config.render_images is True
        assert config.image_format == "svg"
        assert config.renderer == "kroki"
        assert config.theme == "default"


# ============================================================================
# TEST 3: Workflow Creation and Types
# ============================================================================
class TestWorkflowCreation:
    """Test 3: Workflow creation with different execution modes"""

    def test_workflow_types_creation(self):
        """Test creation of different workflow types"""
        from workflow import create_hld_workflow

        sequential = create_hld_workflow("sequential")
        parallel = create_hld_workflow("parallel")
        conditional = create_hld_workflow("conditional")

        assert sequential is not None
        assert parallel is not None
        assert conditional is not None


# ============================================================================
# TEST 4: ML Dataset Generation
# ============================================================================
class TestMLDatasetGeneration:
    """Test 4: ML dataset generation with synthetic data"""

    def test_synthetic_dataset_generator(self):
        """Test synthetic HLD dataset generation"""
        from ml.training.generate_dataset import SyntheticDatasetGenerator

        generator = SyntheticDatasetGenerator(random_state=42)
        df = generator.generate(n_samples=100)  # Use smaller sample for testing

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert "quality_score" in df.columns
        assert len(df.columns) == 38  # 37 features + 1 target (fixed column count)

        # Verify quality score distribution
        assert df['quality_score'].min() >= 0
        assert df['quality_score'].max() <= 100


# ============================================================================
# TEST 5: ML Model Training Pipeline
# ============================================================================
class TestMLModelTraining:
    """Test 5: ML model training with proper data splits"""

    def test_model_training_pipeline(self):
        """Test ML training pipeline initialization"""
        from ml.training.train_large_model import LargeScaleMLTrainer

        trainer = LargeScaleMLTrainer()

        assert trainer is not None
        assert hasattr(trainer, 'load_dataset')
        assert hasattr(trainer, 'prepare_data')
        assert hasattr(trainer, 'train_models')
        assert hasattr(trainer, 'evaluate_models')


# ============================================================================
# TEST 6: ML Quality Prediction
# ============================================================================
class TestMLQualityPrediction:
    """Test 6: ML quality prediction with trained models"""

    def test_quality_predictor_initialization(self):
        """Test HLD quality predictor initialization"""
        from ml.training.inference import HLDQualityPredictor

        predictor = HLDQualityPredictor()

        assert predictor is not None
        assert hasattr(predictor, 'train_models_from_scratch')
        assert hasattr(predictor, 'predict')
        assert hasattr(predictor, 'predict_batch')


# ============================================================================
# TEST 7: Feature Extraction
# ============================================================================
class TestFeatureExtraction:
    """Test 7: ML feature extraction from HLD documents"""

    def test_feature_extractor(self):
        """Test feature extraction from text content"""
        from ml.models.feature_extractor import FeatureExtractor, HLDFeatures

        extractor = FeatureExtractor()

        sample_text = """
        # System Architecture
        This is a comprehensive HLD document with multiple sections.
        ## Authentication
        OAuth2 and JWT are supported.
        ## Database
        PostgreSQL with scaling capabilities.
        """

        features = extractor.extract(sample_text)

        assert isinstance(features, HLDFeatures)
        assert features.word_count > 0
        # Should have text metrics, structure metrics, etc.
        assert hasattr(features, 'word_count') and hasattr(features, 'sentence_count')


# ============================================================================
# TEST 8: Quality Scoring
# ============================================================================
class TestQualityScoring:
    """Test 8: Rule-based quality scoring system"""

    def test_quality_scorer(self):
        """Test rule-based quality scorer"""
        from ml.models.quality_scorer import RuleBasedQualityScorer, QualityScore

        scorer = RuleBasedQualityScorer()

        sample_text = """
        # System Architecture
        This is a comprehensive HLD document with security considerations.
        ## Security Section
        OAuth2 and JWT authentication are implemented.
        ## API Specification
        RESTful endpoints with proper documentation.
        """

        score = scorer.score(sample_text)

        assert isinstance(score, QualityScore)
        assert 0 <= score.overall_score <= 100
        assert 0 <= score.completeness <= 100
        assert 0 <= score.clarity <= 100


# ============================================================================
# TEST 9: Diagram Conversion and Rendering
# ============================================================================
class TestDiagramProcessing:
    """Test 9: Diagram conversion and rendering utilities"""

    def test_diagram_converter(self):
        """Test diagram plan to Mermaid text conversion"""
        from utils.diagram_converter import diagram_plan_to_text

        plan = {
            "class": {
                "nodes": [
                    {"name": "User", "attributes": ["id", "email"]},
                    {"name": "Order", "attributes": ["id", "amount"]}
                ],
                "relations": [
                    {"from": "User", "to": "Order", "type": "1..N"}
                ]
            },
            "sequences": [
                {
                    "title": "Order Flow",
                    "actors": ["User", "System"],
                    "steps": [
                        {"from": "User", "to": "System", "message": "create_order()"}
                    ]
                }
            ]
        }

        result = diagram_plan_to_text(plan)

        assert "class_text" in result
        assert "sequence_texts" in result
        assert len(result["class_text"]) > 0
        assert len(result["sequence_texts"]) > 0


# ============================================================================
# TEST 10: Output Composition
# ============================================================================
class TestOutputComposition:
    """Test 10: HLD document composition and formatting"""

    def test_markdown_composition(self):
        """Test markdown document composition"""
        from utils.compose_output import hld_to_markdown

        # Create mock data
        requirement_name = "TestSystem"
        prd_markdown = "# Test Requirements\nThis is a test document."

        auth_data = {
            'actors': ['User', 'Admin'],
            'flows': ['OAuth2'],
            'idp_options': ['Auth0'],
            'threats': ['CSRF']
        }

        result = hld_to_markdown(
            requirement_name=requirement_name,
            prd_markdown=prd_markdown,
            authentication=auth_data,
            integrations=[],
            entities=[],
            apis=[],
            use_cases=[],
            nfrs={},
            risks=[],
            class_mermaid_text='classDiagram...',
            sequence_mermaid_texts=[],
            hld_base_dir=None
        )

        assert isinstance(result, str)
        assert len(result) > 0
        assert requirement_name.lower() in result.lower()


# ============================================================================
# ADDITIONAL INTEGRATION TESTS
# ============================================================================
class TestMLIntegration:
    """Additional test: ML module integration and imports"""

    def test_ml_modules_import(self):
        """Test that all ML modules can be imported successfully"""
        try:
            from ml.training.generate_dataset import SyntheticDatasetGenerator
            from ml.training.train_large_model import LargeScaleMLTrainer
            from ml.training.inference import HLDQualityPredictor
            from ml.models.feature_extractor import FeatureExtractor
            from ml.models.quality_scorer import RuleBasedQualityScorer

            assert SyntheticDatasetGenerator is not None
            assert LargeScaleMLTrainer is not None
            assert HLDQualityPredictor is not None
            assert FeatureExtractor is not None
            assert RuleBasedQualityScorer is not None
        except ImportError as e:
            pytest.fail(f"ML module import failed: {e}")


class TestUtilityFunctions:
    """Additional test: Utility functions and helpers"""

    def test_utility_imports(self):
        """Test that all utility modules can be imported"""
        try:
            from utils.diagram_converter import diagram_plan_to_text
            from utils.diagram_renderer import render_diagrams
            from utils.compose_output import hld_to_markdown
            from utils.risk_heatmap import generate_risk_heatmap

            assert diagram_plan_to_text is not None
            assert render_diagrams is not None
            assert hld_to_markdown is not None
            assert generate_risk_heatmap is not None
        except ImportError as e:
            pytest.fail(f"Utility import failed: {e}")


class TestErrorHandling:
    """Additional test: Error handling and edge cases"""

    def test_invalid_feature_values(self):
        """Test handling of invalid feature values"""
        from ml.models.feature_extractor import FeatureExtractor, HLDFeatures

        extractor = FeatureExtractor()

        # Test with empty string
        features = extractor.extract("")
        assert isinstance(features, HLDFeatures)
        assert features.word_count == 0

        # Test with very long text
        long_text = "word " * 10000
        features = extractor.extract(long_text)
        assert isinstance(features, HLDFeatures)
        assert features.word_count > 0


# ============================================================================
# PYTEST CONFIGURATION AND HELPERS
# ============================================================================

@pytest.fixture
def sample_hld_data():
    """Fixture providing sample HLD data for tests"""
    return {
        'pdf_path': 'sample.pdf',
        'requirement_name': 'SampleSystem',
        'extracted_markdown': '# Sample HLD\nThis is a test document.',
        'quality_score': 85.5
    }


@pytest.fixture
def temp_output_dir():
    """Fixture providing temporary output directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "-ra",  # Show summary of all test outcomes
        "--color=yes"  # Colored output
    ])
