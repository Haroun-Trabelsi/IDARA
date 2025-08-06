import React from 'react'
import AuthModal from 'components/AuthModal'
import Header from 'components/Header'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import 'styles/ReactWelcome.css'
import About from 'components/About'
import ProjectManagementInterface from 'components/Projects'
import CollaboratorPage from 'components/HeaderComponents/CollaboratorPage'; // Ajouter cette importation
import Login from 'components/HeaderComponents/Login'; // Ajouter cette importation
import Error from 'components/HeaderComponents/error'; // Ajouter cette importation
import Register from 'components/HeaderComponents/Register'; // Ajouter cette importation
import CompletProfil from 'components/HeaderComponents/CompleteProfil'; // Ajouter cette importation
import Account from 'components/HeaderComponents/Account'; // Ajouter cette importation
import Organizations from 'components/HeaderComponents/OrganizationPage'; // Ajouter cette importation
import VerifyInbox from 'components/HeaderComponents/VerifyEmailPage'; // Ajouter cette importation
import ChatbotWidget from 'components/Chatbot/ChatbotWidget'
import DocumentationPage from 'pages/DocumentationPage'
import AdminMessages from 'components/admin/AdminMessages'



const App = () => {
  return (
    <BrowserRouter>
      <div className='min-h-screen flex flex-col bg-gray-900 text-gray-300 w-full'>
        <Header />
        <Routes>
          <Route path="/" element={<ProjectManagementInterface />} />
          <Route path="/collaborator" element={<CollaboratorPage />} /> {/* Nouvelle route */}
          <Route path="/verify-email" element={<VerifyInbox />} /> {/* Nouvelle route */}

          <Route path="/Projects" element={<ProjectManagementInterface />} />
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route path="/CompleteProfil" element={<CompletProfil />} />
          <Route path="/Account" element={<Account />} />
          <Route path="/organizations" element={<Organizations />} />
          <Route path="/support" element={<DocumentationPage />} />

<Route path="/admin/messages" element={<AdminMessages />} />

          <Route path="/error" element={<Error />} />


                
        </Routes>
        <AuthModal />
        <About />
         <ChatbotWidget />
      </div>
    </BrowserRouter>
  )
}

export default App
