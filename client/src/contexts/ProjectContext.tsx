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

const ProjectContext = createContext<ProjectContextType | null>(null);

type ProjectProviderProps = {
  children: ReactNode;
};

export const ProjectProvider = ({ children }: ProjectProviderProps) => {
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);

  return (
    <ProjectContext.Provider value={{ selectedProject, setSelectedProject }}>
      {children}
    </ProjectContext.Provider>
  );
};

export const useProject = () => {
  const context = useContext(ProjectContext);
  if (!context) {
    throw new Error("useProject must be used within a ProjectProvider");
  }
  return context;
};
