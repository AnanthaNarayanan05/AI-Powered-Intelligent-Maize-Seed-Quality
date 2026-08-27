import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import Analyze from "./pages/Analyze";
import Batch from "./pages/Batch";
import History from "./pages/History";
import Copilot from "./pages/Copilot";
import Compare from "./pages/Compare";
import Lot from "./pages/Lot";
import SystemInfo from "./pages/SystemInfo";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/analyze" element={<Analyze />} />
          <Route path="/batch" element={<Batch />} />
          <Route path="/history" element={<History />} />
          <Route path="/copilot" element={<Copilot />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/lot" element={<Lot />} />
          <Route path="/system" element={<SystemInfo />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
