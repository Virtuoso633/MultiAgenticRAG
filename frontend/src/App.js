// frontend/src/App.js (Corrected Query Handling and Enhanced Interrupt UI)
import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './App.css';

function App() {
    const [query, setQuery] = useState('');
    const [messages, setMessages] = useState([]);
    const [isTyping, setIsTyping] = useState(false);
    const [isDarkMode, setIsDarkMode] = useState(false);
    const [interruptData, setInterruptData] = useState(null); // State to hold interrupt data
    const websocket = useRef(null);
    const messagesEndRef = useRef(null);
    const currentResponse = useRef("");

    useEffect(() => {
        websocket.current = new WebSocket('ws://localhost:8000/ws');

        websocket.current.onopen = () => console.log('WebSocket connection opened');

        websocket.current.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                console.log("Frontend received message:", data);

                if (data.error) {
                    setMessages(prev => [...prev, { type: 'error', text: data.error }]);
                    setIsTyping(false);
                }
                else if (data.content) {
                    currentResponse.current += data.content;
                    setMessages(prev => {
                        if (prev.length > 0 && prev[prev.length - 1].type === 'response') {
                            const updated = [...prev];
                            updated[updated.length - 1] = { type: 'response', text: currentResponse.current };
                            return updated;
                        } else {
                            return [...prev, { type: 'response', text: currentResponse.current }];
                        }
                    });
                }
                else if (data.interrupt) {
                    // --- INTERRUPT MESSAGE RECEIVED ---
                    console.log("Interrupt message received from backend:", data.interrupt); // Debug log
                    setInterruptData(data.interrupt); // Store interrupt data in state
                    setMessages(prev => [
                        ...prev,
                        { type: 'llm_output', text: data.interrupt.llm_output }, // Display LLM output related to interrupt
                        { type: 'interrupt', text: data.interrupt.question, binaryScore: data.interrupt.binary_score } // Display the question prompt
                    ]);
                    setIsTyping(false); // Stop typing indicator when interrupt occurs
                }
                else if (data.end) {
                    setIsTyping(false);
                    currentResponse.current = "";
                }
            } catch (error) {
                console.error('Error parsing WebSocket message:', error, event.data);
                setMessages(prev => [...prev, { type: 'error', text: 'Error processing server message.' }]);
                setIsTyping(false);
            }
        };

        websocket.current.onclose = () => console.log('Disconnected from WebSocket');
        websocket.current.onerror = (error) => console.error('WebSocket error:', error);

        return () => {
            if (websocket.current) websocket.current.close();
        };
    }, []);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!query.trim() || !websocket.current) return;

        setIsTyping(true);
        currentResponse.current = "";
        setMessages(prev => [...prev, { type: 'user', text: query }]);
        websocket.current.send(JSON.stringify({ query }));
        setQuery('');
    };

    const handleInterruptResponse = (response) => {
        // --- SENDING RESUME RESPONSE TO BACKEND ---
        console.log("Sending resume response to backend:", response); // Debug log
        if (websocket.current) {
            setIsTyping(true);
            websocket.current.send(JSON.stringify({ resume: response }));
            setMessages(prev => prev.filter(msg => msg.type !== 'interrupt' && msg.type !== 'llm_output')); // Clear interrupt messages
            setInterruptData(null); // Clear interrupt data state
        }
    };

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    return (
        <div className={`App ${isDarkMode ? "dark-mode" : ""}`}>
            <header className="app-header">
                <h1>MultiAgentic RAG Chat</h1>
                <button className="dark-toggle" onClick={() => setIsDarkMode(prev => !prev)} title="Toggle dark mode">
                    {isDarkMode ? "☀️" : "🌙"}
                </button>
            </header>

            <div className="chat-container">
                <div className="messages">
                    {messages.map((msg, index) => (
                        <div key={index} className={`message ${msg.type}`}>
                            {msg.type === 'response' || msg.type === 'llm_output' ? (
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
                            ) : (
                                <>{msg.text}</>
                            )}
                            {msg.type === 'interrupt' && (
                                // --- INTERRUPT UI RENDERING ---
                                // In the interrupt-ui div:
                                <div className="interrupt-ui">
                                    <div className="interrupt-question">{msg.text}</div>
                                    <div className="hallucination-score">Hallucination Score: {msg.binaryScore}</div>
                                    <div className="button-container">
                                        <button onClick={() => handleInterruptResponse('y')}>Yes (Retry)</button>
                                        <button onClick={() => handleInterruptResponse('n')}>No (End)</button>
                                    </div>
                                </div>
                            )}
                        </div>
                    ))}
                    {isTyping && <div className="message typing">Agent is typing...</div>}
                    <div ref={messagesEndRef} />
                </div>

                {/* --- CONDITIONAL INTERRUPT POPUP (Alternative UI - you can choose either this or in-message UI) --- */}
                {/* {interruptData && (
                    <div className="interrupt-popup">
                        <p>{interruptData.question}</p>
                        <p>LLM Output: {interruptData.llm_output}</p>
                        <p>Binary Score: {interruptData.binary_score}</p>
                        <button onClick={() => handleInterruptResponse("y")}>Continue (y)</button>
                        <button onClick={() => handleInterruptResponse("n")}>Stop (n)</button>
                    </div>
                )} */}

                <form onSubmit={handleSubmit}>
                    <input type="text" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Enter your query..." disabled={isTyping} />
                    <button type="submit" disabled={isTyping}>Send</button>
                </form>
            </div>
        </div>
    );
}

export default App;
