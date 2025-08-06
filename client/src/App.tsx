import 'styles/ReactWelcome.css'
import { ProjectProvider } from './contexts/ProjectContext';
import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import AuthModal from 'components/AuthModal';
import Header from 'components/Header';
import 'styles/ReactWelcome.css';
import About from 'components/About';
import ProjectManagementInterface from 'components/Projects';
import CollaboratorPage from 'components/HeaderComponents/CollaboratorPage';
import Login from 'components/HeaderComponents/Login';
import Error from 'components/HeaderComponents/error';
import Register from 'components/HeaderComponents/Register';
import CompletProfil from 'components/HeaderComponents/CompleteProfil';
import Account from 'components/HeaderComponents/Account';
import VerifyInbox from 'components/HeaderComponents/VerifyEmailPage';
import AdminDashboard from 'components/AdminDashboard/AdminDashboard';
import Video from 'components/Project-components/CompressVideo';
import Organizations from 'components/ManageOrganization/OrganizationPage';
import { AuthProvider } from 'contexts/AuthContext';
import AuthGuard from 'utils/AuthGuard';
import RoleGuard from 'utils/RoleGuard';

const DefaultLayout = ({ children }: { children: React.ReactNode }) => (
  <div className="min-h-screen flex flex-col bg-gray-900 text-gray-300 w-full">
    <Header />
    {children}
    <AuthModal />
    <About />
  </div>
);

const AdminLayout = ({ children }: { children: React.ReactNode }) => (
  <div className="min-h-screen bg-gray-900 text-gray-300 w-full">
    {children}
  </div>
);

const App = () => {
  return (
    <ProjectProvider>
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          
          <Route path="/" element={
            <DefaultLayout>
          <ProjectManagementInterface />
          </DefaultLayout>} />

          <Route path="/Projects" element={
            <DefaultLayout>
              <ProjectManagementInterface />
              </DefaultLayout>
            } />
          <Route
            path="/login"
            element={
              <DefaultLayout>
                <Login />
              </DefaultLayout>
            }
          />
          <Route
            path="/register"
            element={
              <DefaultLayout>
                <Register />
              </DefaultLayout>
            }
          />
          <Route
            path="/error"
            element={
              <DefaultLayout>
                <Error />
              </DefaultLayout>
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
            path="/collaborator"
            element={
              <DefaultLayout>
                <AuthGuard>
                  <CollaboratorPage />
                </AuthGuard>
              </DefaultLayout>
            }
          />
          <Route
            path="/verify-email"
            element={
              <DefaultLayout>
                <AuthGuard>
                  <VerifyInbox />
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
            path="/CompleteProfil"
            element={
              <DefaultLayout>
                <AuthGuard>
                  <CompletProfil />
                </AuthGuard>
              </DefaultLayout>
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
        <AuthModal />
        </AuthProvider>
    </BrowserRouter>
    </ProjectProvider>
  )
};

export default App;