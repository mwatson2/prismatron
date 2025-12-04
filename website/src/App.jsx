import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import HomePage from './pages/HomePage'
import MechanicalPage from './pages/MechanicalPage'
import ElectricalPage from './pages/ElectricalPage'
import ComputePage from './pages/ComputePage'
import AlgorithmPage from './pages/AlgorithmPage'
import AudioPage from './pages/AudioPage'
import SoftwarePage from './pages/SoftwarePage'
import ControlAppPage from './pages/ControlAppPage'
import SpecsPage from './pages/SpecsPage'
import BOMPage from './pages/BOMPage'
import LicensePage from './pages/LicensePage'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<HomePage />} />
        <Route path="mechanical" element={<MechanicalPage />} />
        <Route path="electrical" element={<ElectricalPage />} />
        <Route path="compute" element={<ComputePage />} />
        <Route path="algorithm" element={<AlgorithmPage />} />
        <Route path="audio" element={<AudioPage />} />
        <Route path="software" element={<SoftwarePage />} />
        <Route path="control-app" element={<ControlAppPage />} />
        <Route path="specs" element={<SpecsPage />} />
        <Route path="bom" element={<BOMPage />} />
        <Route path="license" element={<LicensePage />} />
      </Route>
    </Routes>
  )
}
