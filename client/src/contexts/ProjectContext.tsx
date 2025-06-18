import React, { createContext, useContext, useState, ReactNode } from 'react';

// 1. Define the type of the project and context value
type Project = {
  id: string;
  name: string;
  // Add more fields if needed
};

type ProjectContextType = {
  selectedProject: Project | null;
  setSelectedProject: (project: Project | null) => void;
};

// 2. Create context with correct initial type
const ProjectContext = createContext<ProjectContextType | null>(null);

// 3. Define props for the provider
type ProjectProviderProps = {
  children: ReactNode;
};

// 4. Create the provider
export const ProjectProvider = ({ children }: ProjectProviderProps) => {
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);

  return (
    <ProjectContext.Provider value={{ selectedProject, setSelectedProject }}>
      {children}
    </ProjectContext.Provider>
  );
};

// 5. Create a typed hook to consume the context
export const useProject = () => {
  const context = useContext(ProjectContext);
  if (!context) {
    throw new Error("useProject must be used within a ProjectProvider");
  }
  return context;
};
