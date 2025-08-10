import 'styles/ReactWelcome.css';
import { ProjectProvider } from './contexts/ProjectContext';
import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import AuthModal from 'components/AuthModal';
import Header from 'components/Header';
import About from 'components/About';
import ProjectManagementInterface from 'components/Projects';
import Login from 'components/Auth/Login';
import Error from 'components/error';
import Register from 'components/Auth/Register';
import CompletProfil from 'components/Auth/CompleteProfil';
import Account from 'components/ManageAccount/Account';
import VerifyInbox from 'components/Auth/VerifyEmailPage';
import AdminDashboard from 'components/AdminDashboard/AdminDashboard';
import Video from 'components/Project-components/CompressVideo';
import Organizations from 'components/ManageOrganization/OrganizationPage';
import { AuthProvider } from 'contexts/AuthContext';
import AuthGuard from 'utils/AuthGuard';
import RoleGuard from 'utils/RoleGuard';
import ChatbotWidget from 'components/Chatbot/ChatbotWidget';
import DocumentationPage from 'pages/DocumentationPage';
import AdminMessages from 'components/admin/AdminMessages';


import ResetPasswordPage from 'components/Auth/ResetPasswordPage';
import CheckEmailPage from 'components/Auth/ResetPwdCheckEmailPage';

import RegisterStaticPage from 'components/Auth/RegisterStaticPage';

const DefaultLayout = ({ children }: { children: React.ReactNode }) => (
  <div className="min-h-screen flex flex-col bg-gray-900 text-gray-300 w-full">
    <Header />
    {children}
    <AuthModal />
    <About /> {/* Contient le footer */}
    <ChatbotWidget /> {/* Ajouté ici pour les pages avec DefaultLayout */}
  </div>
);

const AdminLayout = ({ children }: { children: React.ReactNode }) => (
  <div className="min-h-screen bg-gray-900 text-gray-300 w-full">
    {children}
    <ChatbotWidget /> {/* Ajouté ici pour les pages avec AdminLayout */}
  </div>
);

const PublicLayout = ({ children }: { children: React.ReactNode }) => (
  <div className="min-h-screen flex items-center justify-center bg-gray-900 text-gray-300 w-full">
    {children}
  </div>
);

const App = () => {
  return (
    <ProjectProvider>
      <BrowserRouter>
        <AuthProvider>
          <Routes>
            {/* Routes publiques sans Header, Footer, ni Chatbot */}
            <Route
              path="/login"
              element={
                <PublicLayout>
                  <Login />
                </PublicLayout>
              }
            />
            <Route
              path="/register"
              element={
                <PublicLayout>
                  <Register />
                </PublicLayout>
              }
            />
            <Route
              path="/error"
              element={
                <PublicLayout>
                  <Error />
                </PublicLayout>
              }
            />
<Route
  path="/check-email"
  element={
    <PublicLayout>
      <CheckEmailPage />
    </PublicLayout>
  }
/>


<Route
  path="/RegisterStaticPage"
  element={
    <PublicLayout>
      <RegisterStaticPage />
    </PublicLayout>
  }
/>
<Route
  path="/reset-password/:token"
  element={
    <PublicLayout>
      <ResetPasswordPage />
    </PublicLayout>
  }
/>
            {/* Routes protégées avec AuthGuard (pour utilisateurs connectés) */}
            <Route
              path="/"
              element={
                <DefaultLayout>
                  <AuthGuard>
                    <ProjectManagementInterface />
                  </AuthGuard>
                </DefaultLayout>
              }
            />
            <Route
              path="/Projects"
              element={
                <DefaultLayout>
                  <AuthGuard>
                    <ProjectManagementInterface />
                  </AuthGuard>
                </DefaultLayout>
              }
            />
            <Route
              path="/support"
              element={
                <DefaultLayout>
                  <DocumentationPage />
                </DefaultLayout>
              }
            />
            <Route
              path="/admin/messages"
              element={
                <DefaultLayout>
                  <AdminMessages />
                </DefaultLayout>
              }
            />
            <Route
              path="/verify-email"
              element={
                <PublicLayout>
                  <AuthGuard>
                    <VerifyInbox />
                  </AuthGuard>
                </PublicLayout>
              }
            />
            <Route
              path="/CompleteProfil"
              element={
                <PublicLayout>
                  <AuthGuard>
                    <CompletProfil />
                  </AuthGuard>
                </PublicLayout>
              }
            />
            <Route
              path="/Account"
              element={
                <DefaultLayout>
                  <AuthGuard>
                    <Account />
                  </AuthGuard>
                </DefaultLayout>
              }
            />
            <Route
              path="/organizations"
              element={
                <DefaultLayout>
                  <AuthGuard>
                    <Organizations />
                  </AuthGuard>
                </DefaultLayout>
              }
            />
            <Route
              path="/video"
              element={
                <DefaultLayout>
                  <AuthGuard>
                    <Video />
                  </AuthGuard>
                </DefaultLayout>
              }
            />
            <Route
              path="/AdminDashboard"
              element={
                <AdminLayout>
                  <AuthGuard>
                    <RoleGuard isAdminRoute={true}>
                      <AdminDashboard />
                    </RoleGuard>
                  </AuthGuard>
                </AdminLayout>
              }
            />
          </Routes>
        </AuthProvider>
      </BrowserRouter>
    </ProjectProvider>
  );
};

export default App;