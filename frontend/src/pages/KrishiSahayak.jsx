import React, { useState, useEffect } from "react";
import { useTranslation } from "react-i18next";
import axios from "axios";
import i18n from "i18next";
import Header from "../components/Header";

const KrishiSahayak = () => {
  const { t } = useTranslation();
  const [messages, setMessages] = useState([
    { text: t("Welcome to KrishiSahayak! How can I help you today?"), sender: "bot" }
  ]);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    i18n.changeLanguage(localStorage.getItem("languagePreference") || "en");
  }, []);

  const handleSend = async () => {
    if (!message.trim()) return;

    const userMessage = { text: message, sender: "user" };
    setMessages([...messages, userMessage]);
    setMessage("");
    setLoading(true);

    try {
      const lang = localStorage.getItem("languagePreference") || "en";
      const response = await axios.post("http://127.0.0.1:5000/chatbot", {
        language: lang,
        query: message,
      });

      setMessages((prev) => [
        ...prev,
        { text: response.data || t("I am KrishiSahayak and will answer only farming-related queries."), sender: "bot" },
      ]);
    } catch (error) {
      console.error("Error fetching response:", error);
      setMessages((prev) => [
        ...prev,
        { text: t("Something went wrong. Please try again."), sender: "bot" },
      ]);
    }

    setLoading(false);
  };

  return (
    <div className="min-h-screen flex flex-col">
      <Header name={t("Welcome to KrishiSahayak")} />

      {/* Chat Window */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.map((msg, index) => (
          <div key={index} className={`flex ${msg.sender === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`p-4 max-w-2xl text-lg rounded-lg shadow-md ${
                msg.sender === "user" 
                  ? "bg-green-600 text-white" 
                  : "bg-white text-gray-900 border border-gray-200"
              }`}
              style={{ wordWrap: "break-word", whiteSpace: "pre-wrap" }}
            >
              <pre className="whitespace-pre-wrap">{msg.text}</pre>
            </div>
          </div>
        ))}
        {loading && <p className="text-gray-500">{t("Typing...")}</p>}
      </div>

      {/* Chat Input */}
      <div className="flex items-center p-4 bg-white border-t shadow-md">
        <input
          type="text"
          placeholder={t("Type your farming query...")}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          className="flex-1 p-4 text-lg border border-gray-300 rounded-lg shadow-sm focus:ring-green-500 focus:border-green-500"
        />
        <button 
          onClick={handleSend} 
          className="ml-2 bg-green-600 text-white px-6 py-4 text-lg rounded-lg hover:bg-green-700 transition duration-300 shadow-md"
          disabled={loading}
        >
          {loading ? t("Sending...") : t("Send")}
        </button>
      </div>
    </div>
  );
};

export default KrishiSahayak;
