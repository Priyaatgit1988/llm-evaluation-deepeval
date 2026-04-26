import React, { useState, useRef, useEffect } from 'react';

export default function ChatWindow({ messages, onSend }) {
  const [input, setInput] = useState('');
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    onSend(input.trim());
    setInput('');
  };

  return (
    <div className="chat-window">
      <div className="chat-header">💬 Chat Assistant</div>
      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`msg msg-${msg.role}`}>
            <div className="msg-avatar">{msg.role === 'bot' ? '🤖' : '👤'}</div>
            <div className="msg-bubble">
              {msg.text.split('\n').map((line, j) => (
                <span key={j}>{line}<br /></span>
              ))}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
      <form className="chat-input" onSubmit={handleSubmit}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Ask about products, orders, shipping..."
          aria-label="Chat message input"
        />
        <button type="submit" aria-label="Send message">Send</button>
      </form>
    </div>
  );
}
