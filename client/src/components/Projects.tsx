"use client"

import { Box, TextField, Typography, ThemeProvider, createTheme, CssBaseline } from "@mui/material"
import { useEffect, useState } from "react"
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
interface VideoComponent {
  name: string;
  fileType: string;
  date: Date;
  value: string;
}



export default function ProjectManagementInterface() {
  const [ProjectData, setProjectData] = useState<Task[]>();
  
  const [filterText, setFilterText] = useState("");
  const { selectedProject } = useProject();

useEffect(() => {
  if (!selectedProject) return;

  const fetchProject = async () => {
    try {
      const response = await fetch(
        `http://localhost:8080/api/projects/${encodeURIComponent(selectedProject.id)}`
      );
      const tasks: Task[] = await response.json();

      const tasksWithVideos = await Promise.all(
        tasks.map(async (task) => {
          try {
            const res = await fetch(`http://localhost:8080/api/task/${task.id}/components`);
            const videoList = await res.json();

          const formattedVideos: VideoComponent[] = videoList
          .map((v: any): VideoComponent => ({
            name: v.name,
            fileType: v.fileType,
            date: new Date(v.date),
            value: v.url.value,
          }))
          .reverse();

        const videosWithFormattedDates = formattedVideos.map((v: VideoComponent) => ({
          ...v,
          date: v.date.toLocaleDateString(), // format after sorting
        }));

            return {
              ...task,
              videos: videosWithFormattedDates,
            };
          } catch (err) {
            console.error(`Failed to load videos for task ${task.id}`, err);
            return {
              ...task,
              videos: [],
            };
          }
        })
      );

      console.log("Fetched project tasks with reversed video lists:", tasksWithVideos);
      setProjectData(tasksWithVideos);
    } catch (error) {
      console.error("Failed to fetch project:", error);
    }
  };

  fetchProject();
}, [selectedProject]);



  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box sx={{ display: "flex", flexDirection: "column", height: "94vh" }}>
        {/* Header Component */}

        <Box sx={{ display: "flex", flexGrow: 1, overflow: "hidden" }}>

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

            
            
{Array.isArray(ProjectData) && ProjectData.length > 0 && selectedProject ? (
  <TaskTable
    project={selectedProject.name}
    tasks={ProjectData}
    setTasks={setProjectData}
  />
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
