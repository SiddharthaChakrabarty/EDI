import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './pages/Home'
import Sidebar from './components/Sidebar';
import CropRecommendation from './pages/CropRecommendation';
import CropCategoryPage from './pages/CropCategoryPage';
import CropDetailPage from './pages/CropDetailPage';
import PestChatbot from './pages/PestChatbot';
import KrishiSahayak from './pages/KrishiSahayak';
import WeatherForecast from './pages/WeatherForecast';
import CropCalendarPage from './pages/CropCalendar';
import MarketTrends from './pages/MarketTrends';
import YieldPredictionPage from './pages/YieldPredictionPage';
import AgriculturalLoans from './pages/AgriculturalLoans';
import ComplaintsPage from './pages/ComplaintsPage';
import MicroInvestments from './pages/MicroInvestments';

function App() {
  return (
    <Router>
      <Sidebar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/recommendations" element={<CropRecommendation />} />
        <Route path="/crop/:category" element={<CropCategoryPage />} />
        <Route path="/crop-detail/:name" element={<CropDetailPage />} />
        <Route path="/pest-chatbot" element={<PestChatbot />} />
        <Route path="/chatbot" element={<KrishiSahayak />} />
        <Route path="/weather" element={<WeatherForecast />} />
        <Route path="/crop-calendar" element={<CropCalendarPage />} />
        <Route path="/market-trends" element={<MarketTrends />} />
        <Route path="/yield-pred" element={<YieldPredictionPage />} />
        <Route path="/agri-loans" element={<AgriculturalLoans />} />
        <Route path="/complaints" element={<ComplaintsPage />} />
        <Route path="/investments" element={<MicroInvestments />} />
      </Routes>
    </Router>
  );
}

export default App;
