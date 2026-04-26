import React, { useState } from 'react';
import ChatWindow from './components/ChatWindow';
import ProductCatalog from './components/ProductCatalog';
import './App.css';

const PRODUCTS = [
  { id: 1, name: 'Wireless Headphones', price: 79.99, category: 'Electronics', description: 'Noise-cancelling Bluetooth headphones with 30hr battery', image: '🎧', stock: 45 },
  { id: 2, name: 'Running Shoes', price: 129.99, category: 'Footwear', description: 'Lightweight mesh running shoes with cushioned sole', image: '👟', stock: 30 },
  { id: 3, name: 'Organic Coffee Beans', price: 18.99, category: 'Grocery', description: '1lb bag of single-origin Ethiopian coffee beans', image: '☕', stock: 120 },
  { id: 4, name: 'Yoga Mat', price: 34.99, category: 'Fitness', description: 'Non-slip eco-friendly yoga mat, 6mm thick', image: '🧘', stock: 60 },
  { id: 5, name: 'Laptop Stand', price: 49.99, category: 'Electronics', description: 'Adjustable aluminum laptop stand with ventilation', image: '💻', stock: 25 },
  { id: 6, name: 'Water Bottle', price: 24.99, category: 'Fitness', description: 'Insulated stainless steel bottle, keeps cold 24hrs', image: '🍶', stock: 80 },
  { id: 7, name: 'Backpack', price: 59.99, category: 'Accessories', description: 'Water-resistant 25L backpack with laptop compartment', image: '🎒', stock: 35 },
  { id: 8, name: 'Sunglasses', price: 39.99, category: 'Accessories', description: 'Polarized UV400 sunglasses, unisex design', image: '🕶️', stock: 50 },
];

function generateBotResponse(userMessage, products) {
  const msg = userMessage.toLowerCase();

  if (msg.includes('hello') || msg.includes('hi') || msg.includes('hey')) {
    return "Hello! 👋 Welcome to our e-commerce store. I can help you find products, check prices, availability, or answer questions about orders. What are you looking for today?";
  }
  if (msg.includes('return') || msg.includes('refund')) {
    return "Our return policy allows returns within 30 days of purchase. Items must be unused and in original packaging. Refunds are processed within 5-7 business days. Would you like to start a return?";
  }
  if (msg.includes('shipping') || msg.includes('delivery')) {
    return "We offer free standard shipping on orders over $50 (3-5 business days). Express shipping is $9.99 (1-2 business days). International shipping starts at $14.99.";
  }
  if (msg.includes('track') || msg.includes('order status')) {
    return "To track your order, please provide your order number (format: ORD-XXXXX). You can also check your order status in the 'My Orders' section of your account.";
  }
  if (msg.includes('discount') || msg.includes('coupon') || msg.includes('sale')) {
    return "🎉 Current promotions:\n• WELCOME10 — 10% off your first order\n• SUMMER25 — 25% off fitness items\n• Free shipping on orders over $50\nEnter the code at checkout!";
  }
  if (msg.includes('payment') || msg.includes('pay')) {
    return "We accept Visa, MasterCard, American Express, PayPal, and Apple Pay. All transactions are secured with SSL encryption. Need help with a payment issue?";
  }

  // Product search
  const matchedProducts = products.filter(p =>
    msg.includes(p.name.toLowerCase()) ||
    msg.includes(p.category.toLowerCase()) ||
    p.description.toLowerCase().split(' ').some(word => word.length > 3 && msg.includes(word))
  );

  if (matchedProducts.length > 0) {
    const list = matchedProducts.map(p => `• ${p.image} ${p.name} — $${p.price} (${p.stock} in stock)`).join('\n');
    return `Here's what I found:\n${list}\nWould you like more details on any of these?`;
  }

  if (msg.includes('product') || msg.includes('catalog') || msg.includes('what do you sell')) {
    const categories = [...new Set(products.map(p => p.category))];
    return `We have products in these categories: ${categories.join(', ')}. Browse the catalog on the left, or ask me about a specific product!`;
  }

  if (msg.includes('cheap') || msg.includes('budget') || msg.includes('affordable')) {
    const sorted = [...products].sort((a, b) => a.price - b.price).slice(0, 3);
    const list = sorted.map(p => `• ${p.image} ${p.name} — $${p.price}`).join('\n');
    return `Here are our most affordable items:\n${list}`;
  }

  if (msg.includes('expensive') || msg.includes('premium') || msg.includes('best')) {
    const sorted = [...products].sort((a, b) => b.price - a.price).slice(0, 3);
    const list = sorted.map(p => `• ${p.image} ${p.name} — $${p.price}`).join('\n');
    return `Here are our premium picks:\n${list}`;
  }

  return "I'm not sure I understand. I can help with:\n• Product search & recommendations\n• Pricing & availability\n• Shipping & delivery info\n• Returns & refunds\n• Discounts & promotions\n• Payment methods\nWhat would you like to know?";
}

export default function App() {
  const [messages, setMessages] = useState([
    { role: 'bot', text: "Hi there! 👋 I'm your shopping assistant. Ask me about products, orders, shipping, or anything else!" }
  ]);

  const handleSend = (text) => {
    const userMsg = { role: 'user', text };
    const botReply = { role: 'bot', text: generateBotResponse(text, PRODUCTS) };
    setMessages(prev => [...prev, userMsg, botReply]);
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>🛒 ShopSmart</h1>
        <span>Your AI Shopping Assistant</span>
      </header>
      <div className="app-body">
        <ProductCatalog products={PRODUCTS} />
        <ChatWindow messages={messages} onSend={handleSend} />
      </div>
    </div>
  );
}
