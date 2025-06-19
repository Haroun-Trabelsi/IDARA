"use client"

import { Box, TextField, Typography, ThemeProvider, createTheme, CssBaseline } from "@mui/material"
import { useEffect, useState } from "react"
import Sidebar from "./Project-components/Sidebar"
import SecondaryNavigation from "./Project-components/SecondaryNavigation"
import Toolbar from "./Project-components/Toolbar"
import TaskTable, {  Task } from "./Project-components/TaskTable"
import React from "react"
import { useProject } from '../contexts/ProjectContext';

// Enhanced dark theme matching the design
const darkTheme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: "#4299e1",
    },
    secondary: {
      main: "#f59e0b",
    },
    background: {
      default: "#0f172a",
      paper: "#1a202c",
    },
    text: {
      primary: "#e2e8f0",
      secondary: "#a0aec0",
    },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundColor: "#0f172a",
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: "none",
        },
      },
    },
  },
})



export default function ProjectManagementInterface() {
  const [ProjectData, setProjectData] = useState<Task[]>();
  const [filterText, setFilterText] = useState("");
    const { selectedProject } = useProject();

  useEffect(() => {
    if (!selectedProject) return; // only fetch if projectName exists

    const fetchProject = async () => {
      try {
      const response = await fetch(`http://localhost:8080/api/projects/${encodeURIComponent(selectedProject.id)}`)
      const data = await response.json(); // This should already be in Task[] shape
      console.log("Fetched project tasks:", data);
      setProjectData(data); // Set flat list directly
      } catch (error) {
        console.error("Failed to fetch project:", error);
      }
    };

    fetchProject();
  }, [selectedProject]);

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box sx={{ display: "flex", flexDirection: "column", height: "100vh" }}>
        {/* Header Component */}

        <Box sx={{ display: "flex", flexGrow: 1, overflow: "hidden" }}>
          {/* Sidebar Component */}
          <Sidebar />
          {/* Main Content */}
          <Box sx={{ flexGrow: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
            {/* Secondary Navigation */}
            <SecondaryNavigation />

            {/* Toolbar */}
            <Toolbar />

            {/* Filter Bar */}
            <Box sx={{ borderBottom: "1px solid #2d3748", bgcolor: "#1a202c", px: 2, py: 1.5 }}>
              
              <TextField
                placeholder="Type to filter..."
                value={filterText}
                onChange={(e) => setFilterText(e.target.value)}
                size="small"
                sx={{
                  width: 300,
                  "& .MuiOutlinedInput-root": {
                    bgcolor: "#2d3748",
                    "& fieldset": {
                      borderColor: "#4a5568",
                    },
                    "&:hover fieldset": {
                      borderColor: "#718096",
                    },
                    "&.Mui-focused fieldset": {
                      borderColor: "#4299e1",
                    },
                  },
                  "& .MuiInputBase-input": {
                    color: "#e2e8f0",
                    fontSize: "14px",
                  },
                  "& .MuiInputBase-input::placeholder": {
                    color: "#a0aec0",
                    opacity: 1,
                  },
                }}
              />
            </Box>

            {/* Task Summary */}
            <Box sx={{ borderBottom: "1px solid #2d3748", bgcolor: "#1a202c", px: 2, py: 1.5 }}>
              <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                  <Typography variant="body2" color="#718096" sx={{ fontSize: "13px" }}>
                    Tasks
                  </Typography>
                  <Typography variant="body2" color="#718096" sx={{ fontSize: "13px" }}>
                    Count (task)
                  </Typography>
                  <Typography variant="body2" color="#718096" sx={{ fontSize: "13px" }}>
                   57
                  </Typography>
                </Box>
                <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                  <Typography variant="body2" color="#718096" sx={{ fontSize: "13px" }}>
                    Done %
                  </Typography>
                  <Typography variant="body2" color="#718096" sx={{ fontSize: "13px" }}>
                    84.21%
                  </Typography>
                  <Typography variant="body2" color="#718096" sx={{ fontSize: "13px" }}>
                    Sum
                  </Typography>
                  <Typography variant="body2" color="#718096" sx={{ fontSize: "13px" }}>
                    2024-09-10
                  </Typography>
                  <Typography variant="body2" color="#718096" sx={{ fontSize: "13px" }}>
                    Sum
                  </Typography>
                  <Typography variant="body2" color="#718096" sx={{ fontSize: "13px" }}>
                    344.0
                  </Typography>
                  <Typography variant="body2" color="#718096" sx={{ fontSize: "13px" }}>
                    Sum
                  </Typography>
                  <Typography variant="body2" color="#718096" sx={{ fontSize: "13px" }}>
                    119.88
                  </Typography>
                </Box>
              </Box>
            </Box>

            {/* Task Table Component */}
            
{Array.isArray(ProjectData) && ProjectData.length > 0 ? (
  <TaskTable tasks={ProjectData} />
) : (
  <Typography variant="body2" color="textSecondary">
    No tasks available for this project.
  </Typography>
)}

          </Box>
        </Box>
      </Box>
    </ThemeProvider>
  )
}
