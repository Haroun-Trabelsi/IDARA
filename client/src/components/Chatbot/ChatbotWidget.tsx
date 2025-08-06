import React, { useState, useRef, useEffect } from 'react';
// Removed unused lucide-react imports


interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
}

const ChatbotWidget: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: 'Hello! I\'m your virtual assistant. How can I help you today?',
      isUser: false,
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Remplacez par votre clé API Gemini
  const GEMINI_API_KEY = process.env.REACT_APP_GEMINI_API_KEY || 'AIzaSyBF3bFBJjnNIKwN3VBn6EGpFE6TAdyIorA';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessageToGemini = async (message: string): Promise<string> => {
    try {
      const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=${GEMINI_API_KEY}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          contents: [{
            parts: [{
              text: message
            }]
          }],
          generationConfig: {
            temperature: 0.7,
            topK: 40,
            topP: 0.95,
            maxOutputTokens: 1024,
          }
        })
      });

      if (!response.ok) {
        throw new Error(`Erreur HTTP: ${response.status}`);
      }

      const data = await response.json();
      return data.candidates[0].content.parts[0].text;
    } catch (error) {
      console.error('Erreur lors de l\'appel à l\'API Gemini:', error);
      return 'Désolé, je rencontre des difficultés techniques. Veuillez réessayer plus tard.';
    }
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputValue,
      isUser: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const botResponse = await sendMessageToGemini(inputValue);
      
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: botResponse,
        isUser: false,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Erreur:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  return (
    <>
      {/* Widget Button - Positionné en bas à droite */}
      <div
        onClick={toggleChat}
        style={{
          position: 'fixed',
          bottom: '20px',
          right: '20px',
          width: '60px',
          height: '60px',
          backgroundColor: '#4285f4',
          color: 'white',
          borderRadius: '50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: '24px',
          cursor: 'pointer',
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
          zIndex: 1000,
          transition: 'all 0.3s ease',
          border: 'none'
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.transform = 'scale(1.1)';
          e.currentTarget.style.backgroundColor = '#3367d6';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.transform = 'scale(1)';
          e.currentTarget.style.backgroundColor = '#4285f4';
        }}
      >
        {isOpen ? '✕' : '💬'}
        {!isOpen && (
          <div
            style={{
              position: 'absolute',
              top: '-2px',
              right: '-2px',
              width: '12px',
              height: '12px',
              backgroundColor: '#ff4444',
              borderRadius: '50%',
              animation: 'pulse 2s infinite'
            }}
          />
        )}
      </div>

      {/* Chat Popup */}
      {isOpen && (
        <div
          style={{
            position: 'fixed',
            bottom: '90px',
            right: '20px',
            width: '350px',
            height: '500px',
            backgroundColor: 'white',
            borderRadius: '12px',
            boxShadow: '0 8px 24px rgba(0, 0, 0, 0.3)',
            display: 'flex',
            flexDirection: 'column',
            zIndex: 999,
            overflow: 'hidden',
            border: '1px solid #e0e0e0'
          }}
        >
          {/* Header */}
          <div
            style={{
              padding: '15px',
              backgroundColor: '#4285f4',
              color: 'white',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div
                style={{
                  width: '32px',
                  height: '32px',
                  backgroundColor: 'rgba(255, 255, 255, 0.2)',
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-bot w-5 h-5"><path d="M12 8V4H8"></path><rect width="16" height="12" x="4" y="8" rx="2"></rect><path d="M2 14h2"></path><path d="M20 14h2"></path><path d="M15 13v2"></path><path d="M9 13v2"></path></svg>
              </div>
              <div>
                <h3 style={{ margin: 0, fontSize: '16px', fontWeight: '600' }}>Assistant IA</h3>
                <p style={{ margin: 0, fontSize: '12px', opacity: 0.9 }}>En ligne</p>
              </div>
            </div>
            <button
              onClick={toggleChat}
              style={{
                background: 'none',
                border: 'none',
                color: 'white',
                fontSize: '18px',
                cursor: 'pointer',
                padding: '4px',
                borderRadius: '4px'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.2)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = 'transparent';
              }}
            >
              ✕
            </button>
          </div>

          {/* Messages */}
          <div
            style={{
              flex: 1,
              overflowY: 'auto',
              padding: '15px',
              backgroundColor: '#f9f9f9'
            }}
          >
            {messages.map((message) => (
              <div
                key={message.id}
                style={{
                  marginBottom: '15px',
                  display: 'flex',
                  justifyContent: message.isUser ? 'flex-end' : 'flex-start'
                }}
              >
                <div
                  style={{
                    maxWidth: '80%',
                    padding: '10px 15px',
                    borderRadius: '18px',
                    backgroundColor: message.isUser ? '#4285f4' : '#ffffff',
                    color: message.isUser ? 'white' : '#333',
                    fontSize: '14px',
                    lineHeight: '1.4',
                    boxShadow: '0 1px 2px rgba(0, 0, 0, 0.1)',
                    borderBottomRightRadius: message.isUser ? '4px' : '18px',
                    borderBottomLeftRadius: message.isUser ? '18px' : '4px'
                  }}
                >
                  <p style={{ margin: 0 }}>{message.text}</p>
                  <p
                    style={{
                      margin: '5px 0 0 0',
                      fontSize: '11px',
                      opacity: 0.7
                    }}
                  >
                    {message.timestamp.toLocaleTimeString('fr-FR', {
                      hour: '2-digit',
                      minute: '2-digit'
                    })}
                  </p>
                </div>
              </div>
            ))}
            
            {isLoading && (
              <div style={{ display: 'flex', justifyContent: 'flex-start', marginBottom: '15px' }}>
                <div
                  style={{
                    backgroundColor: '#ffffff',
                    padding: '10px 15px',
                    borderRadius: '18px',
                    borderBottomLeftRadius: '4px',
                    boxShadow: '0 1px 2px rgba(0, 0, 0, 0.1)'
                  }}
                >
                  <div style={{ display: 'flex', gap: '4px' }}>
                    <div
                      style={{
                        width: '8px',
                        height: '8px',
                        backgroundColor: '#ccc',
                        borderRadius: '50%',
                        animation: 'bounce 1.4s ease-in-out infinite both'
                      }}
                    />
                    <div
                      style={{
                        width: '8px',
                        height: '8px',
                        backgroundColor: '#ccc',
                        borderRadius: '50%',
                        animation: 'bounce 1.4s ease-in-out 0.16s infinite both'
                      }}
                    />
                    <div
                      style={{
                        width: '8px',
                        height: '8px',
                        backgroundColor: '#ccc',
                        borderRadius: '50%',
                        animation: 'bounce 1.4s ease-in-out 0.32s infinite both'
                      }}
                    />
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div
            style={{
              padding: '15px',
              borderTop: '1px solid #eee',
              backgroundColor: 'white'
            }}
          >
            <div style={{ display: 'flex', gap: '10px' }}>
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Tapez votre message..."
                disabled={isLoading}
                style={{
                  flex: 1,
                  padding: '10px 15px',
                  border: '1px solid #ddd',
                  borderRadius: '25px',
                  outline: 'none',
                  fontSize: '14px',
                   color: 'black',
                }}
                onFocus={(e) => {
                  e.currentTarget.style.borderColor = '#4285f4';
                  e.currentTarget.style.boxShadow = '0 0 0 2px rgba(66, 133, 244, 0.2)';
                }}
                onBlur={(e) => {
                  e.currentTarget.style.borderColor = '#ddd';
                  e.currentTarget.style.boxShadow = 'none';
                }}
              />
              <button
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || isLoading}
                style={{
                  width: '40px',
                  height: '40px',
                  backgroundColor: !inputValue.trim() || isLoading ? '#ccc' : '#4285f4',
                  color: 'white',
                  border: 'none',
                  borderRadius: '50%',
                  cursor: !inputValue.trim() || isLoading ? 'not-allowed' : 'pointer',
                  fontSize: '16px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  transition: 'background-color 0.2s'
                }}
                onMouseEnter={(e) => {
                  if (!isLoading && inputValue.trim()) {
                    e.currentTarget.style.backgroundColor = '#3367d6';
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isLoading && inputValue.trim()) {
                    e.currentTarget.style.backgroundColor = '#4285f4';
                  }
                }}
              >
                ▶
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Styles d'animation */}
      <style>
        {`
          @keyframes pulse {
            0% { transform: scale(0.95); opacity: 1; }
            70% { transform: scale(1); opacity: 0.7; }
            100% { transform: scale(0.95); opacity: 1; }
          }
          
          @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
          }
        `}
      </style>
    </>
  );
};

export default ChatbotWidget;