"""
DeepEval tests for the E-Commerce Chatbot.
Evaluates chatbot responses using 15+ metrics with configurable judge LLM.
"""
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
    ToxicityMetric,
    BiasMetric,
    GEval,
)

from custom_model import CustomEvalModel
from config import JUDGE_LLM
from test_data import CHATBOT_TEST_CASES

# ─── Initialize Judge Model ───
judge_model = CustomEvalModel(JUDGE_LLM)


# ─── Metric Definitions (15+ metrics) ───

def get_answer_relevancy():
    return AnswerRelevancyMetric(threshold=0.5, model=judge_model)

def get_faithfulness():
    return FaithfulnessMetric(threshold=0.5, model=judge_model)

def get_contextual_precision():
    return ContextualPrecisionMetric(threshold=0.5, model=judge_model)

def get_contextual_recall():
    return ContextualRecallMetric(threshold=0.5, model=judge_model)

def get_contextual_relevancy():
    return ContextualRelevancyMetric(threshold=0.5, model=judge_model)

def get_hallucination():
    return HallucinationMetric(threshold=0.5, model=judge_model)

def get_toxicity():
    return ToxicityMetric(threshold=0.5, model=judge_model)

def get_bias():
    return BiasMetric(threshold=0.5, model=judge_model)

def get_correctness():
    return GEval(
        name="Correctness",
        criteria="Determine if the actual output is factually correct based on the expected output.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5,
        model=judge_model,
    )

def get_coherence():
    return GEval(
        name="Coherence",
        criteria="Evaluate if the response is logically coherent and well-structured.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5,
        model=judge_model,
    )

def get_completeness():
    return GEval(
        name="Completeness",
        criteria="Check if the response covers all key points from the expected output.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5,
        model=judge_model,
    )

def get_conciseness():
    return GEval(
        name="Conciseness",
        criteria="Evaluate if the response is concise without unnecessary verbosity.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5,
        model=judge_model,
    )

def get_helpfulness():
    return GEval(
        name="Helpfulness",
        criteria="Rate how helpful the response is for the customer's query in an e-commerce context.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5,
        model=judge_model,
    )

def get_politeness():
    return GEval(
        name="Politeness",
        criteria="Evaluate if the response maintains a polite and professional customer service tone.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5,
        model=judge_model,
    )

def get_safety():
    return GEval(
        name="Safety",
        criteria="Check that the response does not contain harmful, misleading, or inappropriate content.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5,
        model=judge_model,
    )


# ─── Simulate chatbot response ───
def get_chatbot_response(user_input: str) -> str:
    """Simulate chatbot response (same logic as React chatbot)."""
    msg = user_input.lower()

    if any(w in msg for w in ['hello', 'hi', 'hey', 'help']):
        return "Hello! Welcome to our e-commerce store. I can help you find products, check prices, availability, or answer questions about orders. What are you looking for today?"
    if 'return' in msg or 'refund' in msg:
        return "Our return policy allows returns within 30 days of purchase. Items must be unused and in original packaging. Refunds are processed within 5-7 business days."
    if 'shipping' in msg or 'delivery' in msg:
        return "We offer free standard shipping on orders over $50 (3-5 business days). Express shipping is $9.99 (1-2 business days). International shipping starts at $14.99."
    if 'track' in msg or 'order status' in msg:
        return "To track your order, please provide your order number (format: ORD-XXXXX). You can also check your order status in the My Orders section."
    if 'discount' in msg or 'coupon' in msg or 'sale' in msg:
        return "Current promotions: WELCOME10 for 10% off first order, SUMMER25 for 25% off fitness items, free shipping on orders over $50."
    if 'payment' in msg or 'pay' in msg:
        return "We accept Visa, MasterCard, American Express, PayPal, and Apple Pay. All transactions are secured with SSL encryption."
    if 'headphone' in msg:
        return "Wireless Headphones — $79.99 (45 in stock). Noise-cancelling Bluetooth headphones with 30hr battery."
    if 'cheap' in msg or 'budget' in msg or 'affordable' in msg:
        return "Our most affordable items: Organic Coffee Beans $18.99, Water Bottle $24.99, Yoga Mat $34.99."
    if 'fitness' in msg:
        return "Fitness products: Yoga Mat $34.99 (non-slip eco-friendly), Water Bottle $24.99 (insulated stainless steel)."
    if 'categor' in msg:
        return "We have products in: Electronics, Footwear, Grocery, Fitness, Accessories."
    return "I can help with product search, pricing, shipping, returns, discounts, and payment methods."


# ─── Test Functions ───

@pytest.mark.parametrize("test_case", CHATBOT_TEST_CASES, ids=[tc["input"][:40] for tc in CHATBOT_TEST_CASES])
def test_chatbot_answer_relevancy(test_case):
    actual = get_chatbot_response(test_case["input"])
    tc = LLMTestCase(input=test_case["input"], actual_output=actual, retrieval_context=test_case["context"])
    assert_test(tc, [get_answer_relevancy()])

@pytest.mark.parametrize("test_case", CHATBOT_TEST_CASES, ids=[tc["input"][:40] for tc in CHATBOT_TEST_CASES])
def test_chatbot_faithfulness(test_case):
    actual = get_chatbot_response(test_case["input"])
    tc = LLMTestCase(input=test_case["input"], actual_output=actual, retrieval_context=test_case["context"])
    assert_test(tc, [get_faithfulness()])

@pytest.mark.parametrize("test_case", CHATBOT_TEST_CASES, ids=[tc["input"][:40] for tc in CHATBOT_TEST_CASES])
def test_chatbot_hallucination(test_case):
    actual = get_chatbot_response(test_case["input"])
    tc = LLMTestCase(input=test_case["input"], actual_output=actual, context=test_case["context"])
    assert_test(tc, [get_hallucination()])

@pytest.mark.parametrize("test_case", CHATBOT_TEST_CASES, ids=[tc["input"][:40] for tc in CHATBOT_TEST_CASES])
def test_chatbot_toxicity(test_case):
    actual = get_chatbot_response(test_case["input"])
    tc = LLMTestCase(input=test_case["input"], actual_output=actual)
    assert_test(tc, [get_toxicity()])

@pytest.mark.parametrize("test_case", CHATBOT_TEST_CASES, ids=[tc["input"][:40] for tc in CHATBOT_TEST_CASES])
def test_chatbot_bias(test_case):
    actual = get_chatbot_response(test_case["input"])
    tc = LLMTestCase(input=test_case["input"], actual_output=actual)
    assert_test(tc, [get_bias()])

@pytest.mark.parametrize("test_case", CHATBOT_TEST_CASES, ids=[tc["input"][:40] for tc in CHATBOT_TEST_CASES])
def test_chatbot_contextual_precision(test_case):
    actual = get_chatbot_response(test_case["input"])
    tc = LLMTestCase(
        input=test_case["input"], actual_output=actual,
        expected_output=test_case["expected_output"], retrieval_context=test_case["context"]
    )
    assert_test(tc, [get_contextual_precision()])

@pytest.mark.parametrize("test_case", CHATBOT_TEST_CASES, ids=[tc["input"][:40] for tc in CHATBOT_TEST_CASES])
def test_chatbot_contextual_recall(test_case):
    actual = get_chatbot_response(test_case["input"])
    tc = LLMTestCase(
        input=test_case["input"], actual_output=actual,
        expected_output=test_case["expected_output"], retrieval_context=test_case["context"]
    )
    assert_test(tc, [get_contextual_recall()])

@pytest.mark.parametrize("test_case", CHATBOT_TEST_CASES, ids=[tc["input"][:40] for tc in CHATBOT_TEST_CASES])
def test_chatbot_contextual_relevancy(test_case):
    actual = get_chatbot_response(test_case["input"])
    tc = LLMTestCase(input=test_case["input"], actual_output=actual, retrieval_context=test_case["context"])
    assert_test(tc, [get_contextual_relevancy()])

@pytest.mark.parametrize("test_case", CHATBOT_TEST_CASES, ids=[tc["input"][:40] for tc in CHATBOT_TEST_CASES])
def test_chatbot_correctness(test_case):
    actual = get_chatbot_response(test_case["input"])
    tc = LLMTestCase(input=test_case["input"], actual_output=actual, expected_output=test_case["expected_output"])
    assert_test(tc, [get_correctness()])

@pytest.mark.parametrize("test_case", CHATBOT_TEST_CASES, ids=[tc["input"][:40] for tc in CHATBOT_TEST_CASES])
def test_chatbot_coherence(test_case):
    actual = get_chatbot_response(test_case["input"])
    tc = LLMTestCase(input=test_case["input"], actual_output=actual)
    assert_test(tc, [get_coherence()])

@pytest.mark.parametrize("test_case", CHATBOT_TEST_CASES, ids=[tc["input"][:40] for tc in CHATBOT_TEST_CASES])
def test_chatbot_completeness(test_case):
    actual = get_chatbot_response(test_case["input"])
    tc = LLMTestCase(input=test_case["input"], actual_output=actual, expected_output=test_case["expected_output"])
    assert_test(tc, [get_completeness()])

@pytest.mark.parametrize("test_case", CHATBOT_TEST_CASES, ids=[tc["input"][:40] for tc in CHATBOT_TEST_CASES])
def test_chatbot_conciseness(test_case):
    actual = get_chatbot_response(test_case["input"])
    tc = LLMTestCase(input=test_case["input"], actual_output=actual)
    assert_test(tc, [get_conciseness()])

@pytest.mark.parametrize("test_case", CHATBOT_TEST_CASES, ids=[tc["input"][:40] for tc in CHATBOT_TEST_CASES])
def test_chatbot_helpfulness(test_case):
    actual = get_chatbot_response(test_case["input"])
    tc = LLMTestCase(input=test_case["input"], actual_output=actual)
    assert_test(tc, [get_helpfulness()])

@pytest.mark.parametrize("test_case", CHATBOT_TEST_CASES, ids=[tc["input"][:40] for tc in CHATBOT_TEST_CASES])
def test_chatbot_politeness(test_case):
    actual = get_chatbot_response(test_case["input"])
    tc = LLMTestCase(input=test_case["input"], actual_output=actual)
    assert_test(tc, [get_politeness()])

@pytest.mark.parametrize("test_case", CHATBOT_TEST_CASES, ids=[tc["input"][:40] for tc in CHATBOT_TEST_CASES])
def test_chatbot_safety(test_case):
    actual = get_chatbot_response(test_case["input"])
    tc = LLMTestCase(input=test_case["input"], actual_output=actual)
    assert_test(tc, [get_safety()])
