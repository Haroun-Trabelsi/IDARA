import React, { useState } from 'react';
import { Book, Search, ChevronRight, CheckCircle, AlertCircle, Info, Clock, BookOpen } from 'lucide-react';

const DocumentationPage = () => {
  const [activeSection, setActiveSection] = useState('getting-started');
  const [searchTerm, setSearchTerm] = useState('');

  // Documentation data
  const docSections = [
    {
      id: 'getting-started',
      title: 'Quick Start',
      icon: '🚀',
      description: 'Set up IDara in a few simple steps',
      content: [
        {
          title: '1. First Login',
          type: 'steps',
          items: [
            { text: 'Receive your credentials via email', status: 'info' },
            { text: 'Log in to your IDara workspace', status: 'info' },
            { text: 'Configure your Ftrack API key', status: 'warning' },
            { text: 'Import your first project', status: 'success' }
          ]
        },
        {
          title: '2. Initial Setup',
          type: 'steps',
          items: [
            { text: 'Add your Ftrack instance URL', status: 'info' },
            { text: 'Test the connection', status: 'warning' },
            { text: 'Invite your collaborators', status: 'info' },
            { text: 'Set up permissions', status: 'success' }
          ]
        }
      ]
    },
    {
      id: 'ftrack-integration',
      title: 'Ftrack Integration',
      icon: '🔗',
      description: 'Connect IDara to your Ftrack Studio instance',
      content: [
        {
          title: 'Getting your Ftrack API Key',
          type: 'guide',
          description: 'Follow these steps to retrieve your Ftrack API key:',
          items: [
            { text: 'Log in to your Ftrack Studio', status: 'info' },
            { text: 'Go to Settings > API Keys', status: 'info' },
            { text: 'Click on "Create API Key"', status: 'warning' },
            { text: 'Name your key (e.g., "IDara Integration")', status: 'info' },
            { text: 'Copy the generated key', status: 'success' },
            { text: 'Paste it in IDara > Settings > Integrations', status: 'success' }
          ]
        },
        {
          title: 'Ftrack URL Configuration',
          type: 'code',
          description: 'Format of your Ftrack instance URL:',
          code: 'https://your-studio.ftrackapp.com',
          items: [
            { text: 'Replace "your-studio" with your studio name', status: 'warning' },
            { text: 'Verify that the URL is accessible', status: 'info' },
            { text: 'Test the connection in IDara', status: 'success' }
          ]
        }
      ]
    },
    {
      id: 'video-ai',
      title: 'AI Video Analysis',
      icon: '🤖',
      description: 'Use AI to analyze your video content',
      content: [
        {
          title: 'Supported Video Formats',
          type: 'list',
          items: [
            { text: 'MP4 (recommended)', status: 'success' },
            { text: 'MOV (Apple)', status: 'success' },
            { text: 'AVI (Windows)', status: 'info' },
            { text: 'MKV (Matroska)', status: 'info' },
            { text: 'WEBM (Web)', status: 'info' }
          ]
        },
        {
          title: 'Limits and Constraints',
          type: 'warning',
          items: [
            { text: 'Maximum size: 2GB per file', status: 'warning' },
            { text: 'Maximum duration: 2 hours', status: 'warning' },
            { text: 'Minimum resolution: 480p', status: 'info' },
            { text: 'Analysis time: 5-15 minutes', status: 'info' }
          ]
        }
      ]
    }
  ];

  const getStatusIcon = (status) => {
    switch (status) {
      case 'success': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'warning': return <AlertCircle className="w-4 h-4 text-yellow-500" />;
      case 'info': return <Info className="w-4 h-4 text-blue-500" />;
      default: return <Info className="w-4 h-4 text-gray-500" />;
    }
  };

  const getStatusBg = (status) => {
    switch (status) {
      case 'success': return 'bg-green-50 border-green-200';
      case 'warning': return 'bg-yellow-50 border-yellow-200';
      case 'info': return 'bg-blue-50 border-blue-200';
      default: return 'bg-gray-50 border-gray-200';
    }
  };

  const filteredSections = docSections.filter(section =>
    section.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
    section.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
    section.content.some(item =>
      item.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.items?.some(step => step.text.toLowerCase().includes(searchTerm.toLowerCase()))
    )
  );

  const activeContent = docSections.find(section => section.id === activeSection);

  return (
    <div className="min-h-screen bg-gray-50" style={{minHeight: '100vh', backgroundColor: '#f9fafb'}}>
      {/* Header */}
      <div className="bg-white shadow-sm border-b" style={{backgroundColor: 'white', borderBottom: '1px solid #e5e7eb', boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)'}}>
        <div className="max-w-7xl mx-auto px-4 py-6" style={{maxWidth: '80rem', margin: '0 auto', padding: '24px 16px'}}>
          <div className="flex items-center justify-between" style={{display: 'flex', alignItems: 'center', justifyContent: 'space-between'}}>
            <div className="flex items-center space-x-4" style={{display: 'flex', alignItems: 'center', gap: '16px'}}>
              <div className="p-2 bg-blue-600 rounded-lg shadow-lg" style={{padding: '8px', backgroundColor: '#2563eb', borderRadius: '8px', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)'}}>
                <Book className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900" style={{fontSize: '24px', fontWeight: 'bold', color: '#111827'}}>
                  IDara Documentation
                </h1>
                <p className="text-gray-600 font-medium" style={{color: '#4b5563', fontWeight: '500'}}>Complete guide to get started</p>
              </div>
            </div>
            
            {/* Search */}
            <div className="relative max-w-md w-full" style={{position: 'relative', maxWidth: '28rem', width: '100%'}}>
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" style={{position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: '#9ca3af'}} />
              <input
                type="text"
                placeholder="Search documentation..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                style={{
                  width: '100%',
                  paddingLeft: '40px',
                  paddingRight: '16px',
                  paddingTop: '12px',
                  paddingBottom: '12px',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  fontSize: '14px'
                }}
              />
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8" style={{maxWidth: '80rem', margin: '0 auto', padding: '32px 16px'}}>
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8" style={{display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '32px'}}>
          
          {/* Sidebar Navigation */}
          <div className="lg:col-span-1" style={{gridColumn: 'span 1'}}>
            <div className="bg-white rounded-lg shadow-sm border p-6 sticky top-24" style={{
              backgroundColor: 'white',
              borderRadius: '8px',
              boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)',
              border: '1px solid #e5e7eb',
              padding: '24px',
              position: 'sticky',
              top: '96px'
            }}>
              <div className="flex items-center space-x-2 mb-6" style={{display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '24px'}}>
                <BookOpen className="w-5 h-5 text-blue-600" />
                <h3 className="font-bold text-gray-900" style={{fontWeight: 'bold', color: '#111827'}}>Sections</h3>
              </div>
              <nav className="space-y-3" style={{display: 'flex', flexDirection: 'column', gap: '12px'}}>
                {filteredSections.map((section) => (
                  <button
                    key={section.id}
                    onClick={() => setActiveSection(section.id)}
                    className={`w-full text-left p-4 rounded-lg transition-all flex items-center space-x-3 ${
                      activeSection === section.id
                        ? 'bg-blue-500 text-white shadow-lg'
                        : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900 bg-gray-50'
                    }`}
                    style={{
                      width: '100%',
                      textAlign: 'left',
                      padding: '16px',
                      borderRadius: '8px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '12px',
                      border: 'none',
                      cursor: 'pointer',
                      backgroundColor: activeSection === section.id ? '#3b82f6' : '#f9fafb',
                      color: activeSection === section.id ? 'white' : '#4b5563',
                      boxShadow: activeSection === section.id ? '0 10px 15px -3px rgba(0, 0, 0, 0.1)' : 'none'
                    }}
                  >
                    <span style={{fontSize: '20px'}}>{section.icon}</span>
                    <div className="flex-1 min-w-0" style={{flex: '1', minWidth: '0'}}>
                      <div className="font-semibold text-sm truncate" style={{
                        fontWeight: '600',
                        fontSize: '14px',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap'
                      }}>
                        {section.title}
                      </div>
                      <div className="text-xs opacity-75 truncate" style={{
                        fontSize: '12px',
                        opacity: '0.75',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap'
                      }}>
                        {section.description}
                      </div>
                    </div>
                    <ChevronRight className="w-4 h-4" />
                  </button>
                ))}
              </nav>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3" style={{gridColumn: 'span 3'}}>
            {activeContent && (
              <div className="bg-white rounded-lg shadow-sm border p-8" style={{
                backgroundColor: 'white',
                borderRadius: '8px',
                boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1)',
                border: '1px solid #e5e7eb',
                padding: '32px'
              }}>
                <div className="flex items-center space-x-4 mb-8" style={{display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '32px'}}>
                  <div className="p-3 bg-blue-500 rounded-lg shadow-lg" style={{
                    padding: '12px',
                    backgroundColor: '#3b82f6',
                    borderRadius: '8px',
                    boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)'
                  }}>
                    <span style={{fontSize: '24px'}}>{activeContent.icon}</span>
                  </div>
                  <div>
                    <h2 className="text-3xl font-bold text-gray-900" style={{fontSize: '30px', fontWeight: 'bold', color: '#111827'}}>
                      {activeContent.title}
                    </h2>
                    <p className="text-gray-600 font-medium mt-1" style={{color: '#4b5563', fontWeight: '500', marginTop: '4px'}}>
                      {activeContent.description}
                    </p>
                  </div>
                </div>

                <div className="space-y-10" style={{display: 'flex', flexDirection: 'column', gap: '40px'}}>
                  {activeContent.content.map((item, index) => (
                    <div key={index} className="relative" style={{position: 'relative'}}>
                      <div className="absolute left-0 top-0 bottom-0 w-1 bg-blue-500 rounded-full" style={{
                        position: 'absolute',
                        left: '0',
                        top: '0',
                        bottom: '0',
                        width: '4px',
                        backgroundColor: '#3b82f6',
                        borderRadius: '9999px'
                      }}></div>
                      <div className="pl-8" style={{paddingLeft: '32px'}}>
                        <h3 className="text-xl font-bold text-gray-800 mb-4" style={{fontSize: '20px', fontWeight: 'bold', color: '#1f2937', marginBottom: '16px'}}>
                          {item.title}
                        </h3>
                        
                        {item.description && (
                          <p className="text-gray-600 mb-6 bg-gray-50 p-4 rounded-lg border" style={{
                            color: '#4b5563',
                            marginBottom: '24px',
                            backgroundColor: '#f9fafb',
                            padding: '16px',
                            borderRadius: '8px',
                            border: '1px solid #e5e7eb'
                          }}>
                            {item.description}
                          </p>
                        )}

                        {item.code && (
                          <div className="bg-gray-900 text-green-400 p-6 rounded-lg font-mono text-sm mb-6 shadow-lg" style={{
                            backgroundColor: '#111827',
                            color: '#4ade80',
                            padding: '24px',
                            borderRadius: '8px',
                            fontFamily: 'monospace',
                            fontSize: '14px',
                            marginBottom: '24px',
                            boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)'
                          }}>
                            <div className="flex items-center space-x-2 mb-3" style={{display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px'}}>
                              <div className="w-3 h-3 bg-red-500 rounded-full" style={{width: '12px', height: '12px', backgroundColor: '#ef4444', borderRadius: '50%'}}></div>
                              <div className="w-3 h-3 bg-yellow-500 rounded-full" style={{width: '12px', height: '12px', backgroundColor: '#eab308', borderRadius: '50%'}}></div>
                              <div className="w-3 h-3 bg-green-500 rounded-full" style={{width: '12px', height: '12px', backgroundColor: '#22c55e', borderRadius: '50%'}}></div>
                            </div>
                            {item.code}
                          </div>
                        )}

                        {item.type === 'steps' && (
                          <div className="space-y-4" style={{display: 'flex', flexDirection: 'column', gap: '16px'}}>
                            {item.items.map((step, stepIndex) => (
                              <div key={stepIndex} className={`flex items-start space-x-4 p-4 rounded-lg border ${getStatusBg(step.status)}`} style={{
                                display: 'flex',
                                alignItems: 'flex-start',
                                gap: '16px',
                                padding: '16px',
                                borderRadius: '8px',
                                border: '1px solid',
                                borderColor: step.status === 'success' ? '#bbf7d0' : step.status === 'warning' ? '#fef3c7' : '#dbeafe',
                                backgroundColor: step.status === 'success' ? '#f0fdf4' : step.status === 'warning' ? '#fffbeb' : '#eff6ff'
                              }}>
                                <div className="bg-blue-500 text-white text-sm font-bold px-3 py-2 rounded-full min-w-[32px] text-center shadow-md" style={{
                                  backgroundColor: '#3b82f6',
                                  color: 'white',
                                  fontSize: '14px',
                                  fontWeight: 'bold',
                                  paddingLeft: '12px',
                                  paddingRight: '12px',
                                  paddingTop: '8px',
                                  paddingBottom: '8px',
                                  borderRadius: '9999px',
                                  minWidth: '32px',
                                  textAlign: 'center',
                                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                                }}>
                                  {stepIndex + 1}
                                </div>
                                <div className="flex items-center space-x-3 flex-1" style={{display: 'flex', alignItems: 'center', gap: '12px', flex: '1'}}>
                                  {getStatusIcon(step.status)}
                                  <span className="text-gray-700 leading-relaxed font-medium" style={{color: '#374151', lineHeight: '1.625', fontWeight: '500'}}>
                                    {step.text}
                                  </span>
                                </div>
                              </div>
                            ))}
                          </div>
                        )}

                        {(item.type === 'guide' || item.type === 'list' || item.type === 'warning' || item.type === 'code') && (
                          <div className="space-y-3" style={{display: 'flex', flexDirection: 'column', gap: '12px'}}>
                            {item.items?.map((listItem, listIndex) => (
                              <div key={listIndex} className={`flex items-center space-x-4 p-4 rounded-lg border ${getStatusBg(listItem.status)}`} style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '16px',
                                padding: '16px',
                                borderRadius: '8px',
                                border: '1px solid',
                                borderColor: listItem.status === 'success' ? '#bbf7d0' : listItem.status === 'warning' ? '#fef3c7' : '#dbeafe',
                                backgroundColor: listItem.status === 'success' ? '#f0fdf4' : listItem.status === 'warning' ? '#fffbeb' : '#eff6ff'
                              }}>
                                {getStatusIcon(listItem.status)}
                                <span className="text-gray-700 leading-relaxed font-medium flex-1" style={{color: '#374151', lineHeight: '1.625', fontWeight: '500', flex: '1'}}>
                                  {listItem.text}
                                </span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Footer */}
                <div className="flex justify-between items-center mt-12 pt-8 border-t border-gray-200" style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginTop: '48px',
                  paddingTop: '32px',
                  borderTop: '1px solid #e5e7eb'
                }}>
                  <div className="flex items-center space-x-2 text-sm text-gray-500" style={{display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px', color: '#6b7280'}}>
                    <BookOpen className="w-4 h-4" />
                    <span>Step-by-step guide for IDara</span>
                  </div>
                  <div className="flex items-center space-x-2 text-sm text-gray-500" style={{display: 'flex', alignItems: 'center', gap: '8px', fontSize: '14px', color: '#6b7280'}}>
                    <Clock className="w-4 h-4" />
                    <span>Last updated: {new Date().toLocaleDateString('en-US')}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DocumentationPage;