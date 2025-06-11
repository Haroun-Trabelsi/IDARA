import React from 'react'
import AuthModal from 'components/AuthModal'
import Header from 'components/tmp/Header'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import 'styles/ReactWelcome.css'
import Home from 'components/Home'
import About from 'components/About'
import ProjectManagementInterface from 'components/Projects'

const App = () => {
  return (
    <BrowserRouter>
      <div className='min-h-screen flex flex-col bg-gray-900 text-gray-300 w-full'>
        <Header />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/Projects" element={<ProjectManagementInterface />} />
        </Routes>
        <AuthModal />
        <About />
      </div>
    </BrowserRouter>
  )
}

export default App
