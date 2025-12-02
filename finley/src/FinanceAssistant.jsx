import React, { useState, useEffect, useRef } from 'react';
import { Send, DollarSign, TrendingUp, PieChart, Zap, Bot, User, Menu, X, AlertCircle } from 'lucide-react';
import './FinanceAssistant.css';

const FinanceAssistant = () => {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I\'m Finley, your AI financial assistant. I can help you log expenses, set budgets, check your balance, and provide spending summaries. How can I help you today?', timestamp: new Date() }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [balance, setBalance] = useState(0);
  const [budgets, setBudgets] = useState({});
  const [recentExpenses, setRecentExpenses] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [userStats, setUserStats] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);

  // Particles animation
  const [particles, setParticles] = useState([]);

  // API base URL - configure this based on your setup
  const API_BASE = 'http://localhost:8000';

  useEffect(() => {
    const newParticles = Array.from({ length: 50 }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      delay: Math.random() * 15,
      duration: 10 + Math.random() * 10,
    }));
    setParticles(newParticles);
  }, []);

  // Check health and load initial data
  useEffect(() => {
    const initializeApp = async () => {
      try {
        setError(null);
        // Check backend health
        const healthResponse = await fetch(`${API_BASE}/health`);
        const healthData = await healthResponse.json();
        
        if (healthData.ok) {
          setConnectionStatus('connected');
          
          // Load user stats
          const statsResponse = await fetch(`${API_BASE}/users/demo_user/stats`);
          const statsData = await statsResponse.json();
          
          setUserStats(statsData);
          setBalance(parseFloat(statsData.balance));
          setBudgets(statsData.budgets);
          setRecentExpenses(statsData.recent_transactions);
        }
      } catch (error) {
        console.error('Error initializing app:', error);
        setConnectionStatus('disconnected');
        setError('Unable to connect to backend server. Make sure it\'s running on port 8000.');
      }
    };

    initializeApp();
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  const sendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage = {
      role: 'user',
      content: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/v1/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: 'demo_user',
          message: inputMessage
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // Update state with response
      setBalance(parseFloat(data.balance));
      setBudgets(data.budgets);
      setRecentExpenses(data.recent_expenses);

      const assistantMessage = {
        role: 'assistant',
        content: data.reply,
        timestamp: new Date(),
        intent: data.intent,
        entities: data.entities
      };

      setMessages(prev => [...prev, assistantMessage]);
      setConnectionStatus('connected');
    } catch (error) {
      console.error('Error sending message:', error);
      setConnectionStatus('error');
      setError('Failed to send message. Check backend connection.');
      
      const errorMessage = {
        role: 'assistant',
        content: 'I apologize, but I\'m having trouble connecting to the server. Please check that the backend is running on port 8000 and try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatTime = (date) => {
    return new Intl.DateTimeFormat('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    }).format(date);
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown';
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    } catch {
      return 'Unknown';
    }
  };

  const formatAmount = (amount) => {
    const num = typeof amount === 'string' ? parseFloat(amount) : amount;
    return isNaN(num) ? '0.00' : num.toFixed(2);
  };

  const getConnectionStatusText = () => {
    switch (connectionStatus) {
      case 'connected': return 'Online • Ready to help';
      case 'connecting': return 'Connecting...';
      case 'disconnected': return 'Offline';
      case 'error': return 'Connection error';
      default: return 'Unknown';
    }
  };

  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return '#10b981';
      case 'connecting': return '#f59e0b';
      case 'disconnected': return '#6b7280';
      case 'error': return '#ef4444';
      default: return '#6b7280';
    }
  };

  const quickActions = [
    { text: "What's my balance?", icon: DollarSign },
    { text: "I spent $15 on food", icon: TrendingUp },
    { text: "Set budget for entertainment to $300", icon: PieChart },
    { text: "Show spending summary", icon: Zap }
  ];

  // Calculate balance change (simplified calculation)
  const getBalanceChange = () => {
    if (!userStats || !userStats.category_spending) return '+0.0%';
    
    const totalSpending = Object.values(userStats.category_spending).reduce((sum, val) => sum + val, 0);
    return `Total spent: $${formatAmount(totalSpending)}`;
  };

  return (
    <div className="finance-assistant">
      {/* Animated Background */}
      <div className="animated-background" />
      
      {/* Floating Particles */}
      <div className="particles-container">
        {particles.map(particle => (
          <div
            key={particle.id}
            className="particle"
            style={{
              left: `${particle.x}%`,
              animation: `float ${particle.duration}s infinite linear`,
              animationDelay: `${particle.delay}s`
            }}
          />
        ))}
      </div>

      {/* Error Banner */}
      {error && (
        <div className="error-banner glass-morphism">
          <AlertCircle size={20} />
          <span>{error}</span>
          <button onClick={() => setError(null)} className="error-close">×</button>
        </div>
      )}

      {/* Mobile Menu Button */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className="mobile-menu-button glass-morphism"
      >
        {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
      </button>

      <div className="main-layout">
        {/* Sidebar */}
        <div className={`sidebar glass-morphism ${sidebarOpen ? 'open' : 'closed'}`}>
          <div className="sidebar-content">
            {/* Header */}
            <div className="sidebar-header">
              <div className="bot-avatar gradient-cyan-blue">
                <Bot size={32} />
              </div>
              <h1 className="sidebar-title text-gradient-cyan-blue">
                Finley AI
              </h1>
              <p className="sidebar-subtitle">Your Financial Assistant</p>
            </div>

            {/* Balance Card */}
            <div className="balance-card glass-morphism">
              <div className="balance-header">
                <span className="balance-label">Current Balance</span>
                <DollarSign color="#06b6d4" size={20} />
              </div>
              <div className="balance-amount">
                ${balance.toLocaleString()}
              </div>
              <div className="balance-change">{getBalanceChange()}</div>
            </div>

            {/* Budgets */}
            <div className="budgets-card glass-morphism">
              <h3 className="budgets-header">
                <PieChart color="#a855f7" size={18} style={{ marginRight: '0.5rem' }} />
                Active Budgets
              </h3>
              <div className="budgets-list">
                {Object.entries(budgets).length > 0 ? (
                  Object.entries(budgets).map(([category, amount]) => (
                    <div key={category} className="budget-item">
                      <span className="budget-category">{category}</span>
                      <span className="budget-amount">${formatAmount(amount)}</span>
                    </div>
                  ))
                ) : (
                  <div className="budget-item">
                    <span className="budget-category">No budgets set</span>
                    <span className="budget-amount">$0</span>
                  </div>
                )}
              </div>
            </div>

            {/* Recent Expenses */}
            <div className="activity-card glass-morphism">
              <h3 className="activity-header">
                <TrendingUp color="#10b981" size={18} style={{ marginRight: '0.5rem' }} />
                Recent Activity
              </h3>
              <div className="activity-list">
                {recentExpenses.length > 0 ? (
                  recentExpenses.slice(0, 3).map((expense, index) => (
                    <div key={index} className="activity-item">
                      <div>
                        <span>{expense.receiver || 'Transaction'}</span>
                        <small className="activity-date">{formatDate(expense.date)}</small>
                      </div>
                      <span className={`activity-expense ${expense.type === 'income' ? 'income' : 'expense'}`}>
                        {expense.type === 'income' ? '+' : '-'}${formatAmount(expense.amount)}
                      </span>
                    </div>
                  ))
                ) : (
                  <div className="activity-item">
                    <span>No recent activity</span>
                    <span className="activity-expense">$0.00</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Overlay for mobile */}
        {sidebarOpen && (
          <div 
            className="sidebar-overlay"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        {/* Main Chat Area */}
        <div className="chat-container">
          {/* Chat Header */}
          <div className="chat-header glass-morphism">
            <div className="chat-header-content">
              <div className="chat-header-left">
                <div className="chat-avatar gradient-cyan-blue">
                  <Bot size={20} />
                </div>
                <div className="chat-info">
                  <h2>Finley Assistant</h2>
                  <p>{getConnectionStatusText()}</p>
                </div>
              </div>
              <div className="status-indicator">
                <div 
                  className="status-dot" 
                  style={{ backgroundColor: getConnectionStatusColor() }}
                ></div>
                <span className="status-text">
                  {connectionStatus.charAt(0).toUpperCase() + connectionStatus.slice(1)}
                </span>
              </div>
            </div>
          </div>

          {/* Messages */}
          <div 
            ref={chatContainerRef}
            className="messages-container"
          >
            {messages.map((message, index) => (
              <div
                key={index}
                className={`message-wrapper ${message.role}`}
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className={`message-content ${message.role}`}>
                  <div className={`message-avatar ${message.role === 'user' ? 'gradient-cyan-blue' : 'gradient-purple-pink'}`}>
                    {message.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                  </div>
                  <div className={`message-bubble glass-morphism ${message.role}`}>
                    <div className="message-text">{message.content}</div>
                    <div className="message-time">
                      {formatTime(message.timestamp)}
                    </div>
                  </div>
                </div>
              </div>
            ))}
            
            {isLoading && (
              <div className="message-wrapper assistant">
                <div className="message-content assistant">
                  <div className="message-avatar gradient-purple-pink">
                    <Bot size={16} />
                  </div>
                  <div className="message-bubble glass-morphism assistant">
                    <div className="loading-indicator">
                      <div className="loading-dot"></div>
                      <div className="loading-dot"></div>
                      <div className="loading-dot"></div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Quick Actions */}
          <div className="quick-actions">
            <div className="quick-actions-list">
              {quickActions.map((action, index) => (
                <button
                  key={index}
                  onClick={() => setInputMessage(action.text)}
                  className="quick-action-button glass-morphism"
                >
                  <action.icon size={14} />
                  <span>{action.text}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Input */}
          <div className="input-area glass-morphism">
            <div className="input-container">
              <div className="input-wrapper">
                <textarea
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyDown={handleKeyPress}
                  placeholder="Ask me about your finances..."
                  className="message-input glass-morphism"
                  rows="1"
                  disabled={isLoading}
                />
                <button
                  onClick={sendMessage}
                  disabled={!inputMessage.trim() || isLoading}
                  className="send-button gradient-cyan-blue"
                >
                  <Send size={16} />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FinanceAssistant;