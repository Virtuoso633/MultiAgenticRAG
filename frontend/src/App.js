import React, { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import "./App.css";

function App() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [interruptData, setInterruptData] = useState(null);
  const [sources, setSources] = useState([]); // Add state for sources
  const websocket = useRef(null);
  const messagesEndRef = useRef(null);
  const currentResponse = useRef("");

  useEffect(() => {
    websocket.current = new WebSocket("ws://localhost:8000/ws");
    window.myWebSocket = websocket.current; // For debugging
    console.log("WebSocket assigned:", window.myWebSocket);

    websocket.current.onopen = () => console.log("Connected to WebSocket");

    websocket.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("Received:", data);

      if (data.error) {
        setMessages((prev) => [...prev, { type: "error", text: data.error }]);
        setIsTyping(false);
      } else if (data.content) {
        currentResponse.current += data.content;
        setMessages((prev) => {
          if (prev.length > 0 && prev[prev.length - 1].type === "response") {
            const updated = [...prev];
            updated[updated.length - 1] = {
              type: "response",
              text: currentResponse.current,
              id: prev[prev.length - 1].id || Date.now(), // Ensure ID is preserved
            };
            return updated;
          } else {
            return [
              ...prev,
              {
                type: "response",
                text: currentResponse.current,
                id: Date.now(),
              },
            ];
          }
        });
      } else if (data.sources) {
        // Store document sources when backend sends them
        setSources(data.sources);
      } else if (data.type === "interrupt") {
        console.log("Interrupt received in frontend:", data);
        setInterruptData(data.data);
        setMessages((prev) => [
          ...prev,
          {
            type: "llm_output",
            text: data.data.llm_output || "Potential issue detected",
          },
          {
            type: "interrupt",
            text: data.data.question,
            binaryScore: data.data.binary_score,
          },
        ]);
      } else if (data.end) {
        setIsTyping(false);
        currentResponse.current = "";
      }
    };

    websocket.current.onclose = () =>
      console.log("Disconnected from WebSocket");
    websocket.current.onerror = (error) =>
      console.error("WebSocket error:", error);

    return () => {
      if (websocket.current) websocket.current.close();
    };
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!query.trim() || !websocket.current) return;

    setIsTyping(true);
    currentResponse.current = "";
    setMessages((prev) => [...prev, { type: "user", text: query }]);
    websocket.current.send(JSON.stringify({ query }));
    setQuery("");
  };

  const handleInterruptResponse = (response) => {
    if (websocket.current) {
      setIsTyping(true);
      // Send the response using the "resume" key so that backend can pick it up
      websocket.current.send(JSON.stringify({ resume: response }));
      setMessages((prev) =>
        prev.filter(
          (msg) => msg.type !== "interrupt" && msg.type !== "llm_output"
        )
      );
      setInterruptData(null);
    }
  };

  const submitFeedback = (messageId, feedbackValue) => {
    if (websocket.current) {
      websocket.current.send(
        JSON.stringify({
          feedback: feedbackValue,
          message_id: messageId,
        })
      );
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <>
      <div className={`App ${isDarkMode ? "dark-mode" : ""}`}>
        <header className="app-header">
          <h1>RAGHive</h1>
          <button
            className="dark-toggle"
            onClick={() => setIsDarkMode((prev) => !prev)}
            title="Toggle dark mode"
          >
            {isDarkMode ? "☀️" : "🌙"}
          </button>
        </header>

        <div className="chat-container">
          <div className="messages">
            {messages.map((msg, index) => (
              <Message
                key={index}
                message={msg}
                sources={sources}
                websocketRef={websocket} // Pass the websocket reference
                onFeedback={(value) => msg.id && submitFeedback(msg.id, value)}
                parentHandleInterrupt={handleInterruptResponse} // Pass the interrupt handler
              />
            ))}
            {isTyping && <div className="message typing">Agent is typing...</div>}
            <div ref={messagesEndRef} />
          </div>

          {interruptData && (
            <div className="interrupt-popup">
              <h3>Potential Issue Detected</h3>
              <div className="interrupt-content">
                <p>
                  <strong>System Output:</strong> {interruptData.llm_output}
                </p>
                <p>
                  <strong>Confidence Score:</strong>{" "}
                  {interruptData.binary_score === "1" ? "High" : "Low"}
                </p>
                <p>{interruptData.question}</p>
              </div>
              <div className="interrupt-actions">
                <button onClick={() => handleInterruptResponse("y")}>
                  Continue
                </button>
                <button onClick={() => handleInterruptResponse("n")}>Stop</button>
              </div>
            </div>
          )}

          <form onSubmit={handleSubmit}>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter your query..."
              disabled={isTyping}
            />
            <button type="submit" disabled={isTyping}>
              Send
            </button>
          </form>
        </div>
      </div>
      <ToastContainer position="bottom-right" />
    </>
  );
}

function Message({ message, sources, onFeedback, websocketRef, parentHandleInterrupt }) {
  const [showSources, setShowSources] = useState(false);
  const [showFeedback, setShowFeedback] = useState(false);

  // Extract citation numbers if present
  const citations = (message.text.match(/\[\d+\]/g) || []).map((c) =>
    c.replace(/[[\]]/g, "")  // Fixed escape character (removed backslash)
  );

  // Only show the Sources button for response messages that have citations
  const shouldShowSourcesButton =
    message.type === "response" && citations.length > 0;

  // Only show feedback option for response messages
  const shouldShowFeedbackButton = message.type === "response" && onFeedback;

  // Function to handle regeneration request
  const handleRegeneration = () => {
    if (websocketRef.current) {
      websocketRef.current.send(
        JSON.stringify({
          regenerate: true,
          previous_query: message.text
        })
      );
    }
  };

  // Use the parent's interrupt handler
  const handleInterruptResponse = (response) => {
    if (parentHandleInterrupt) {
      parentHandleInterrupt(response);
    }
  };

  // Extract a summary for the preview
  const getSummary = (text) => {
    const firstSentence = text.split(".")[0];
    return firstSentence.length > 100
      ? firstSentence.substring(0, 100) + "..."
      : firstSentence;
  };

  return (
    <div className={`message ${message.type}`}>
      {message.type === "response" && message.text.length > 200 && (
        <div className="message-summary">
          <strong>Summary:</strong> {getSummary(message.text)}
        </div>
      )}
      {message.type === "response" || message.type === "llm_output" ? (
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {message.text}
        </ReactMarkdown>
      ) : (
        <>{message.text}</>
      )}
      {message.type === "interrupt" && (
        <>
          <div>Hallucination Score: {message.binaryScore}</div>
          <div className="button-container">
            <button onClick={() => handleInterruptResponse("y")}>Yes</button>
            <button onClick={() => handleInterruptResponse("n")}>No</button>
          </div>
        </>
      )}
      {/* Message footer with actions */}
      {(shouldShowSourcesButton || shouldShowFeedbackButton) && (
        <div className="message-footer">
          {shouldShowSourcesButton && (
            <button
              onClick={() => setShowSources(!showSources)}
              className="sources-button"
            >
              {showSources ? "Hide Sources" : "View Sources"}
            </button>
          )}

          {shouldShowFeedbackButton && (
            <button
              onClick={() => setShowFeedback(!showFeedback)}
              className="feedback-button"
            >
              {showFeedback ? "Hide Feedback" : "Give Feedback"}
            </button>
          )}
        </div>
      )}
      {/* Sources panel */}
      {showSources && (
        <div className="sources-panel">
          <h4>Sources</h4>
          {citations.map((c) => {
            const sourceIndex = parseInt(c) - 1;
            const sourceDoc = sources[sourceIndex];

            // Create a better source representation
            let sourceText = "Unknown source";
            if (sourceDoc) {
              if (sourceDoc.title) {
                sourceText = sourceDoc.title;
              } else if (sourceDoc.content) {
                // Try to create a meaningful title from content
                const firstLine = sourceDoc.content.split("\n")[0].trim();
                sourceText =
                  firstLine.length > 50
                    ? firstLine.substring(0, 50) + "..."
                    : firstLine;
              } else if (sourceDoc.page_content) {
                // For Langchain Document objects
                const firstLine = sourceDoc.page_content.split("\n")[0].trim();
                sourceText =
                  firstLine.length > 50
                    ? firstLine.substring(0, 50) + "..."
                    : firstLine;
              }
            }

            return (
              <div key={c} className="source-item">
                <strong>Source [{c}]:</strong> {sourceText}
              </div>
            );
          })}
        </div>
      )}

      {/* Feedback panel */}
      {showFeedback && (
        <div className="feedback-panel">
          <h4>Was this response accurate and helpful?</h4>
          <div className="feedback-buttons">
            <button
              onClick={() => {
                onFeedback("helpful");
                setShowFeedback(false);
                // Use toast notification
                toast.success("Thank you! Your feedback helps improve our system.");
              }}
              className="feedback-button-positive"
            >
              👍 Yes, this was helpful
            </button>
            <button
              onClick={() => {
                onFeedback("not_helpful");
                setShowFeedback(false);
                // Use window.confirm to avoid ESLint error
                if (
                  window.confirm(
                    "Would you like us to try regenerating a better response?"
                  )
                ) {
                  handleRegeneration();
                }
              }}
              className="feedback-button-negative"
            >
              👎 No, needs improvement
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;