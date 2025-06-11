"use client"

import { Box, TextField, Typography, ThemeProvider, createTheme, CssBaseline } from "@mui/material"
import { useState } from "react"
import Sidebar from "./tmp/Sidebar"
import SecondaryNavigation from "./tmp/SecondaryNavigation"
import Toolbar from "./tmp/Toolbar"
import TaskTable, { type Task } from "./tmp/TaskTable"
import React from "react"

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

const mockTasks: Task[] = [
  {
    id: "7760",
    number: 1,
    type: "Project",
    status: "in-progress",
    dueDate: "2024-09-10",
    bidHours: 344.0,
    actualHours: 119.88,
    level: 0,
    expanded: true,
    children: [
      {
        id: "129",
        number: 2,
        type: "Sequence",
        status: "in-progress",
        bidHours: 4.0,
        actualHours: 0.97,
        level: 1,
        expanded: true,
        children: [
          {
            id: "060",
            number: 3,
            type: "Shot",
            status: "in-progress",
            bidHours: 4.0,
            actualHours: 0.97,
            level: 2,
            expanded: true,
            children: [
              {
                id: "tracking-1",
                number: 4,
                type: "Task (CamTrack)",
                status: "completed",
                assignee: "Sami Brahem, Ta...",
                description: "Focal length : 35mm / Camera model : MONSTRO 8K VV",
                bidHours: 4.0,
                actualHours: 0.97,
                level: 3,
                icon: "tracking",
              },
            ],
          },
        ],
      },
      {
        id: "131",
        number: 5,
        type: "Sequence",
        status: "in-progress",
        dueDate: "2024-09-03",
        bidHours: 23.0,
        actualHours: 1.05,
        level: 1,
        expanded: true,
        children: [
          {
            id: "014",
            number: 6,
            type: "Shot",
            status: "in-progress",
            dueDate: "2024-08-30",
            bidHours: 8.0,
            actualHours: 0.02,
            level: 2,
            expanded: true,
            children: [
              {
                id: "tracking-2",
                number: 7,
                type: "Task (CamTrack)",
                status: "completed",
                assignee: "Taeib Gastli",
                dueDate: "2024-08-30",
                bidHours: 8.0,
                actualHours: 0.02,
                level: 3,
                icon: "tracking",
              },
            ],
          },
          {
            id: "015",
            number: 8,
            type: "Shot",
            status: "in-progress",
            dueDate: "2024-09-03",
            bidHours: 15.0,
            actualHours: 1.02,
            level: 2,
            expanded: true,
            children: [
              {
                id: "tracking-3",
                number: 9,
                type: "Task (CamTrack)",
                status: "completed",
                assignee: "Taeib Gastli",
                dueDate: "2024-09-03",
                bidHours: 15.0,
                actualHours: 1.02,
                level: 3,
                icon: "tracking",
              },
            ],
          },
        ],
      },
      {
        id: "132",
        number: 10,
        type: "Sequence",
        status: "pending",
        dueDate: "2024-09-04",
        bidHours: 38.0,
        actualHours: 32.89,
        level: 1,
        expanded: true,
        children: [
          {
            id: "040",
            number: 11,
            type: "Shot",
            status: "omitted",
            bidHours: 4.0,
            actualHours: 4.0,
            level: 2,
            expanded: true,
            children: [
              {
                id: "tracking-4",
                number: 12,
                type: "Task (CamTrack)",
                status: "omitted",
                bidHours: 4.0,
                actualHours: 4.0,
                level: 3,
                icon: "tracking",
              },
            ],
          },
          {
            id: "050",
            number: 13,
            type: "Shot",
            status: "in-progress",
            bidHours: 5.0,
            actualHours: 5.0,
            level: 2,
            expanded: true,
            children: [
              {
                id: "tracking-5",
                number: 14,
                type: "Task (CamTrack)",
                status: "completed",
                assignee: "Taeib Gastli",
                description: "Focal length : 65mm / Camera model : Red MONSTRO 8K VV",
                bidHours: 5.0,
                actualHours: 5.0,
                level: 3,
                icon: "tracking",
              },
            ],
          },
        ],
      },
    ],
  },
]

export default function ProjectManagementInterface() {
  const [filterText, setFilterText] = useState("")

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
            <TaskTable tasks={mockTasks} />
          </Box>
        </Box>
      </Box>
    </ThemeProvider>
  )
}
