// import logo from './logo.svg';
// import './App.css';

// function App() {
//   return (
//     <div className="App">
//       <header className="App-header">
//         <img src={logo} className="App-logo" alt="logo" />
//         <p>
//           Edit <code>src/App.js</code> and save to reload.
//         </p>
//         <a
//           className="App-link"
//           href="https://reactjs.org"
//           target="_blank"
//           rel="noopener noreferrer"
//         >
//           Learn React
//         </a>
//       </header>
//     </div>
//   );
// }

// export default App;

// frontend/src/App.js (Corrected Query Handling)
import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './App.css';

function App() {
    const [query, setQuery] = useState('');
    const [messages, setMessages] = useState([]);
    const websocket = useRef(null);
    const [isTyping, setIsTyping] = useState(false);
    const messagesEndRef = useRef(null);
    const currentResponse = useRef("");  // Accumulate response here

    useEffect(() => {
        websocket.current = new WebSocket('ws://localhost:8000/ws');

        websocket.current.onopen = () => {
            console.log('Connected to WebSocket');
        };

        websocket.current.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log("Received:", data);

            if (data.error) {
                setMessages(prevMessages => [...prevMessages, { type: 'error', text: data.error }]);
                setIsTyping(false);
            } else if (data.content) {
                currentResponse.current += data.content;
                 // Update the last message if it's a response, otherwise add a new one
                setMessages(prevMessages => {
                    if (prevMessages.length > 0 && prevMessages[prevMessages.length - 1].type === 'response') {
                        const updatedMessages = [...prevMessages];
                        updatedMessages[updatedMessages.length - 1] = { type: 'response', text: currentResponse.current };
                        return updatedMessages;
                    } else {
                        return [...prevMessages, { type: 'response', text: currentResponse.current }];
                    }
                });
            } else if (data.tool_calls) {
                //Optionally display to the user, currently we will just log it.
                console.log("Tool Calls:", data.tool_calls);
            } else if (data.queries) {
                // We won't show the queries
                console.log("Queries:", data.queries)
            }
            else if (data.interrupt) {
                setMessages(prevMessages => [...prevMessages,
                    { type: 'llm_output', text: data.interrupt.llm_output },
                    { type: 'interrupt', text: data.interrupt.question, binaryScore: data.interrupt.binary_score } // Store binaryScore
                ]);

            } else if (data.end) {
                setIsTyping(false);
                currentResponse.current = ""; // Reset for next response
            }
        };

        websocket.current.onclose = () => {
            console.log('Disconnected from WebSocket');
        };

        websocket.current.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        return () => {
            if (websocket.current) {
                websocket.current.close();
            }
        };
    }, []);


    const handleSubmit = (e) => {
        e.preventDefault();
        if (!query.trim() || !websocket.current) return;

        setIsTyping(true);
        // CRITICAL: Clear the currentResponse when a new query is submitted.
        currentResponse.current = "";
        setMessages(prevMessages => [...prevMessages, { type: 'user', text: query }]);
        websocket.current.send(query);
        setQuery('');
    };


    const handleInterruptResponse = (response) => {
        if (websocket.current) {
            setIsTyping(true);
            websocket.current.send(response);
            setMessages(messages => messages.filter(msg => msg.type !== 'interrupt' && msg.type !== 'llm_output'));
        }
    };

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }, [messages]);

    return (
        <div className="App">
            <h1>MultiAgentic RAG Chat</h1>
            <div className="chat-container">
                <div className="messages">
                    {messages
                        .filter(msg => ['user', 'response', 'error', 'interrupt', 'llm_output'].includes(msg.type))
                        .map((msg, index) => (
                            <div key={index} className={`message ${msg.type}`}>
                                {msg.type === 'response' || msg.type === 'llm_output' ? (
                                    <ReactMarkdown remarkPlugins={[remarkGfm]} children={msg.text} />
                                ) : (
                                    <>{msg.text}</>
                                )}
                                {msg.type === 'interrupt' && (
                                    <>
                                        <div>Hallucination Score: {msg.binaryScore}</div>
                                        <div className='button-container'>
                                            <button onClick={() => handleInterruptResponse('y')}>Yes</button>
                                            <button onClick={() => handleInterruptResponse('n')}>No</button>
                                        </div>
                                    </>
                                )}
                            </div>
                        ))
                    }
                    {isTyping && <div className="message typing">Agent is typing...</div>}
                    <div ref={messagesEndRef} />
                </div>
                <form onSubmit={handleSubmit}>
                    <input
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Enter your query..."
                        disabled={isTyping}
                    />
                    <button type="submit" disabled={isTyping}>Send</button>
                </form>
            </div>
        </div>
    );
}

export default App;