"""
Test data for evaluating both the e-commerce chatbot and the RAG Explorer.
Contains test cases with inputs, expected outputs, contexts, and retrieval contexts.
"""

# ─── E-Commerce Chatbot Test Cases ───
CHATBOT_TEST_CASES = [
    {
        "input": "Hello, I need help",
        "expected_output": "Welcome greeting with offer to help with products, orders, or shipping",
        "context": ["The chatbot should greet users warmly and offer assistance"],
    },
    {
        "input": "What is your return policy?",
        "expected_output": "Returns allowed within 30 days, items must be unused and in original packaging, refunds processed in 5-7 business days",
        "context": ["Return policy allows returns within 30 days of purchase. Items must be unused and in original packaging. Refunds processed within 5-7 business days."],
    },
    {
        "input": "How much does shipping cost?",
        "expected_output": "Free standard shipping on orders over $50 (3-5 days), express $9.99 (1-2 days), international from $14.99",
        "context": ["Free standard shipping on orders over $50, 3-5 business days. Express shipping $9.99, 1-2 business days. International shipping starts at $14.99."],
    },
    {
        "input": "Do you have any discounts?",
        "expected_output": "Current promotions include WELCOME10 for 10% off first order, SUMMER25 for 25% off fitness items, free shipping over $50",
        "context": ["WELCOME10 gives 10% off first order. SUMMER25 gives 25% off fitness items. Free shipping on orders over $50."],
    },
    {
        "input": "What payment methods do you accept?",
        "expected_output": "Visa, MasterCard, American Express, PayPal, and Apple Pay with SSL encryption",
        "context": ["We accept Visa, MasterCard, American Express, PayPal, and Apple Pay. All transactions secured with SSL encryption."],
    },
    {
        "input": "I want to track my order",
        "expected_output": "Provide order number in format ORD-XXXXX or check My Orders section",
        "context": ["To track orders, provide order number (ORD-XXXXX format) or check My Orders section in account."],
    },
    {
        "input": "Show me headphones",
        "expected_output": "Wireless Headphones at $79.99 with noise-cancelling and 30hr battery",
        "context": ["Wireless Headphones: $79.99, noise-cancelling Bluetooth with 30hr battery, 45 in stock."],
    },
    {
        "input": "What are your cheapest products?",
        "expected_output": "Most affordable items listed by price",
        "context": ["Organic Coffee Beans $18.99, Water Bottle $24.99, Yoga Mat $34.99 are among the cheapest."],
    },
    {
        "input": "Tell me about your fitness products",
        "expected_output": "Yoga Mat $34.99 and Water Bottle $24.99 available in fitness category",
        "context": ["Fitness category includes Yoga Mat ($34.99, non-slip eco-friendly) and Water Bottle ($24.99, insulated stainless steel)."],
    },
    {
        "input": "What categories do you have?",
        "expected_output": "Electronics, Footwear, Grocery, Fitness, Accessories",
        "context": ["Product categories: Electronics, Footwear, Grocery, Fitness, Accessories."],
    },
]

# ─── RAG Explorer Test Cases ───
RAG_TEST_CASES = [
    {
        "input": "What is the return policy?",
        "expected_output": "Customers can return items within 30 days of purchase. Items must be unused, unworn, and in original packaging with all tags attached. Refunds are processed within 5-7 business days.",
        "retrieval_context": [
            "Our return policy allows customers to return items within 30 days of purchase. Items must be unused, unworn, and in their original packaging with all tags attached. Refunds are processed within 5-7 business days after we receive the returned item."
        ],
    },
    {
        "input": "How does the loyalty program work?",
        "expected_output": "Earn 1 point per dollar spent. 100 points equals $5 reward. Members get early access to sales, exclusive discounts, and free shipping. Double points during birthday month.",
        "retrieval_context": [
            "Join our ShopSmart Rewards program to earn points on every purchase. Earn 1 point per dollar spent. 100 points equals a $5 reward. Members get early access to sales, exclusive discounts, and free shipping on all orders. Birthday month bonus: earn double points."
        ],
    },
    {
        "input": "What shipping options are available?",
        "expected_output": "Standard shipping free over $50 (3-5 days), express $9.99 (1-2 days), international from $14.99 (7-14 days). All orders include tracking.",
        "retrieval_context": [
            "Standard shipping takes 3-5 business days and is free on orders over $50. Express shipping costs $9.99 and delivers within 1-2 business days. International shipping starts at $14.99 and takes 7-14 business days."
        ],
    },
    {
        "input": "What is the warranty on electronics?",
        "expected_output": "All electronics come with a 1-year manufacturer warranty covering defects in materials and workmanship. Extended warranty plans available for up to 3 years.",
        "retrieval_context": [
            "All electronic products sold on ShopSmart come with a minimum 1-year manufacturer warranty. This covers defects in materials and workmanship under normal use. Extended warranty plans are available for purchase at checkout, offering coverage for up to 3 years."
        ],
    },
    {
        "input": "Do you offer price matching?",
        "expected_output": "Yes, price match guarantee within 14 days of purchase if you find the same product cheaper at an authorized retailer. Does not apply to clearance or auction items.",
        "retrieval_context": [
            "We offer a price match guarantee within 14 days of purchase. If you find the same product at a lower price from an authorized retailer, we will match that price and refund the difference."
        ],
    },
    {
        "input": "How do I contact customer support?",
        "expected_output": "Available Mon-Fri 9AM-9PM EST, Sat-Sun 10AM-6PM EST. Reach via live chat, email support@shopsmart.com, or phone 1-800-SHOP-SMART.",
        "retrieval_context": [
            "Our customer support team is available Monday through Friday, 9 AM to 9 PM EST, and Saturday-Sunday, 10 AM to 6 PM EST. You can reach us via live chat, email at support@shopsmart.com, or phone at 1-800-SHOP-SMART."
        ],
    },
    {
        "input": "What payment methods are accepted?",
        "expected_output": "Visa, MasterCard, American Express, Discover, PayPal, Apple Pay, Google Pay, Shop Pay. Installment payments via Affirm for orders over $100.",
        "retrieval_context": [
            "We accept all major credit cards including Visa, MasterCard, American Express, and Discover. We also accept PayPal, Apple Pay, Google Pay, and Shop Pay. For orders over $100, we offer installment payments through Affirm."
        ],
    },
    {
        "input": "Is my data safe with ShopSmart?",
        "expected_output": "Yes, ShopSmart uses encrypted payment processing through PCI-compliant processors. Personal data is never sold to third parties. Account data deletion available on request.",
        "retrieval_context": [
            "ShopSmart takes your privacy seriously. We collect only necessary information. We never sell personal data to third parties. All payment information is encrypted and processed through PCI-compliant payment processors."
        ],
    },
    {
        "input": "Can I order in bulk?",
        "expected_output": "Yes, bulk orders of 10+ units get 5-20% volume discounts. Contact business@shopsmart.com for custom quotes. Ships within 5-7 business days.",
        "retrieval_context": [
            "For bulk orders of 10 or more units, we offer volume discounts ranging from 5% to 20% depending on the quantity and product category. Contact our business sales team at business@shopsmart.com."
        ],
    },
    {
        "input": "What is ShopSmart's sustainability commitment?",
        "expected_output": "Uses recycled packaging, carbon offset program for shipping, partners with eco-friendly brands, aiming for 50% sustainable product catalog by 2025.",
        "retrieval_context": [
            "ShopSmart is committed to sustainability. We use recycled packaging materials for all shipments. Our carbon offset program neutralizes the environmental impact of shipping. We partner with eco-friendly brands."
        ],
    },
]
