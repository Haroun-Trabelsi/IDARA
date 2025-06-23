import React from 'react'
import AuthModal from 'components/AuthModal'
import Header from 'components/Header'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import 'styles/ReactWelcome.css'
import { ProjectProvider } from './contexts/ProjectContext';
import ProjectManagementInterface from 'components/Projects'

const App = () => {
  return (
    <ProjectProvider>
    <BrowserRouter>
      <div className='min-h-screen flex flex-col bg-gray-900 text-gray-300 w-full'>
        <Header />
        <Routes>
          <Route path="/" element={<ProjectManagementInterface />} />
          <Route path="/Projects" element={<ProjectManagementInterface />} />
        </Routes>
        <AuthModal />
      </div>
    </BrowserRouter>
    </ProjectProvider>
  )
}

export default App
