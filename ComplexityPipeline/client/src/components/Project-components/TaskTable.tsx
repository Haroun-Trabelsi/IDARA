"use client"

import { useState } from "react"
import {
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  Chip,
  Typography,
  Box,
  IconButton,
  Checkbox,
  Button,
} from "@mui/material"
import {
  KeyboardArrowDown,
  KeyboardArrowRight,
} from "@mui/icons-material"
import React from "react"

export interface Task {
  id: string;
  number: number;
  type: string; // e.g., "Task (CamTrack)"
  status: "completed" | "in-progress" | "omitted" | "pending";
  assignee?: string;
  sequence?: string;
  description?: string;
  dueDate?: string;
  bidHours: number;
  actualHours: number;
  level: number; // Keep if you want to apply styling (e.g., indentation)
  icon?: string;
}


interface TaskTableProps {
  tasks: Task[]
}

export default function TaskTable({ tasks }: TaskTableProps) {
const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set(["0", "129", "131", "132"]))

  const toggleExpanded = (id: string) => {
    const newExpanded = new Set(expandedItems)
    if (newExpanded.has(id)) {
      newExpanded.delete(id)
    } else {
      newExpanded.add(id)
    }
    setExpandedItems(newExpanded)
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "#22c55e"
      case "in-progress":
        return "#f59e0b"
      case "omitted":
        return "#ef4444"
      case "pending":
        return "#ef4444"
      default:
        return "#6b7280"
    }
  }
const [selectedTasks, setSelectedTasks] = useState<Set<string>>(new Set());

const toggleTaskSelection = (taskId: string) => {
  setSelectedTasks(prev => {
    const updated = new Set(prev);
    updated.has(taskId) ? updated.delete(taskId) : updated.add(taskId);
    return updated;
  });
};

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "completed":
        return <Chip label="Completed" size="small" sx={{ bgcolor: "#22c55e", color: "white", fontSize: "0.75rem" }} />
      case "omitted":
        return <Chip label="Omitted" size="small" sx={{ bgcolor: "#ef4444", color: "white", fontSize: "0.75rem" }} />
      default:
        return null
    }
  }

  /*const getTaskIcon = (type: string, icon?: string) => {
    if (icon === "tracking") return <CheckBox fontSize="small" sx={{ color: "#a0aec0" }} />
    switch (type) {
      case "Project":
        return <Home fontSize="small" sx={{ color: "#a0aec0" }} />
      case "Sequence":
        return <Movie fontSize="small" sx={{ color: "#a0aec0" }} />
      case "Shot":
        return <PhotoCamera fontSize="small" sx={{ color: "#a0aec0" }} />
      default:
        return <FilePresent fontSize="small" sx={{ color: "#a0aec0" }} />
    }
  }*/

  const flattenTasks = (tasks: Task[]): Task[] => {
    const result: Task[] = []

    const addTask = (task: Task) => {
      result.push(task)
    }

    tasks.forEach(addTask)
    return result
  }

  const flatTasks = flattenTasks(tasks)
  const selectedTaskObjects = flatTasks.filter(task => selectedTasks.has(task.id));

  function sendToFunction(selectedTaskObjects: Task[]): void {
    console.log(selectedTaskObjects)
    throw new Error("Function not implemented.")
  }

  return (
    <>
    <TableContainer
      component={Paper}
      sx={{
        flexGrow: 1,
        overflow: "auto",
        bgcolor: "#0f172a",
        "& .MuiPaper-root": {
          backgroundImage: "none",
        },
      }}
    >
      <Table stickyHeader size="small">
        <TableHead>
          <TableRow>
            <TableCell
              align="center"
              sx={{
                width: 60,
                bgcolor: "#1a202c",
                color: "#718096",
                borderBottom: "1px solid #2d3748",
                fontSize: "12px",
              }}
            >
              #
            </TableCell>
            <TableCell
              sx={{
                width: 60,
                bgcolor: "#1a202c",
                borderBottom: "1px solid #2d3748",
              }}
            ></TableCell>
            <TableCell
              sx={{
                bgcolor: "#1a202c",
                color: "#718096",
                borderBottom: "1px solid #2d3748",
                fontSize: "12px",
                fontWeight: 500,
              }}
            >
              Tasks
            </TableCell>
            <TableCell
              sx={{
                width: 120,
                bgcolor: "#1a202c",
                color: "#718096",
                borderBottom: "1px solid #2d3748",
                fontSize: "12px",
                fontWeight: 500,
              }}
            >
              Type
            </TableCell>
            <TableCell
              sx={{
                width: 120,
                bgcolor: "#1a202c",
                color: "#718096",
                borderBottom: "1px solid #2d3748",
                fontSize: "12px",
                fontWeight: 500,
              }}
            >
              Status
            </TableCell>
            <TableCell
              sx={{
                width: 150,
                bgcolor: "#1a202c",
                color: "#718096",
                borderBottom: "1px solid #2d3748",
                fontSize: "12px",
                fontWeight: 500,
              }}
            >
              Assignee
            </TableCell>
            <TableCell
              sx={{
                bgcolor: "#1a202c",
                color: "#718096",
                borderBottom: "1px solid #2d3748",
                fontSize: "12px",
                fontWeight: 500,
              }}
            >
              Description
            </TableCell>
            <TableCell
              sx={{
                width: 120,
                bgcolor: "#1a202c",
                color: "#718096",
                borderBottom: "1px solid #2d3748",
                fontSize: "12px",
                fontWeight: 500,
              }}
            >
              Due date
            </TableCell>
            <TableCell
              align="right"
              sx={{
                width: 100,
                bgcolor: "#1a202c",
                color: "#718096",
                borderBottom: "1px solid #2d3748",
                fontSize: "12px",
                fontWeight: 500,
              }}
            >
              Bid hours
            </TableCell>
            <TableCell
              align="right"
              sx={{
                width: 100,
                bgcolor: "#1a202c",
                color: "#718096",
                borderBottom: "1px solid #2d3748",
                fontSize: "12px",
                fontWeight: 500,
              }}
            >
              +/- hours
            </TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {flatTasks.map((task) => (
            <TableRow
              key={`${task.id}-${task.number}`}
              hover
              sx={{
                "&:hover": {
                  bgcolor: "rgba(148, 163, 184, 0.05)",
                },
              }}
            >
              <TableCell align="center" sx={{ color: "#718096", p: 1, borderBottom: "1px solid #2d3748" }}>
                <Typography variant="body2" sx={{ fontSize: "13px" }}>
                  {task.number}
                </Typography>
              </TableCell>

              <TableCell sx={{ p: 1, borderBottom: "1px solid #2d3748" }}>
  <Checkbox
    checked={selectedTasks.has(task.id)}
    onChange={() => toggleTaskSelection(task.id)}
    sx={{
      color: "#a0aec0",
      '&.Mui-checked': {
        color: "#63b3ed",
      },
      p: 0.5,
    }}
  />
</TableCell>


              <TableCell sx={{ p: 1, borderBottom: "1px solid #2d3748" }}>
                <Box sx={{ display: "flex", alignItems: "center", pl: task.level * 2.5 }}>
                  
                    <IconButton
                      size="small"
                      onClick={() => toggleExpanded(task.id)}
                      sx={{ p: 0.5, mr: 0.5, color: "#a0aec0" }}
                    >
                      {expandedItems.has(task.id) ? (
                        <KeyboardArrowDown fontSize="small" />
                      ) : (
                        <KeyboardArrowRight fontSize="small" />
                      )}
                    </IconButton>
                  
                  <Typography variant="body2" sx={{ color: "#e2e8f0", fontSize: "13px" }}>
                  {task.sequence || ""} / {task.description || ""} / {task.icon || ''}
                  </Typography>
                </Box>
              </TableCell>

              <TableCell sx={{ p: 1, borderBottom: "1px solid #2d3748" }}>
                <Typography variant="body2" sx={{ color: "#a0aec0", fontSize: "13px" }}>
                  {task.type}
                </Typography>
              </TableCell>

              <TableCell sx={{ p: 1, borderBottom: "1px solid #2d3748" }}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Box sx={{ flexGrow: 1 }}>
                    <LinearProgress
                      variant="determinate"
                      value={task.status === "completed" ? 100 : 70}
                      sx={{
                        height: 8,
                        borderRadius: 1,
                        bgcolor: "#2d3748",
                        "& .MuiLinearProgress-bar": {
                          bgcolor: getStatusColor(task.status),
                        },
                      }}
                    />
                  </Box>
                  {getStatusBadge(task.status)}
                </Box>
              </TableCell>

              <TableCell sx={{ p: 1, borderBottom: "1px solid #2d3748" }}>
                <Typography variant="body2" noWrap sx={{ color: "#a0aec0", fontSize: "13px" }}>
                  {task.assignee || ""}
                </Typography>
              </TableCell>

              <TableCell sx={{ p: 1, borderBottom: "1px solid #2d3748" }}>
                <Typography variant="body2" noWrap sx={{ color: "#a0aec0", fontSize: "13px" }}>
                  
                </Typography>
              </TableCell>

              <TableCell sx={{ p: 1, borderBottom: "1px solid #2d3748" }}>
                <Typography variant="body2" sx={{ color: "#a0aec0", fontSize: "13px" }}>
                  {task.dueDate || ""}
                </Typography>
              </TableCell>

              <TableCell align="right" sx={{ p: 1, borderBottom: "1px solid #2d3748" }}>
                <Typography variant="body2" sx={{ color: "#a0aec0", fontSize: "13px" }}>
                  {task.bidHours.toFixed(2)}
                </Typography>
              </TableCell>

              <TableCell align="right" sx={{ p: 1, borderBottom: "1px solid #2d3748" }}>
                <Typography variant="body2" sx={{ color: "#a0aec0", fontSize: "13px" }}>
                  {task.actualHours.toFixed(2)}
                </Typography>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>

    </TableContainer>
    <Box
  sx={{
    position: "fixed",
    bottom: 24,
    left: "50%",
    transform: "translateX(-50%)",
    zIndex: 10,
  }}
>
  <Button
    variant="contained"
    color="primary"
    size="medium"
    disabled={selectedTasks.size === 0}
    onClick={() => sendToFunction(selectedTaskObjects)}
    sx={{
      px: 4,
      py: 1.5,
      fontSize: "14px",
      borderRadius: 2,
      boxShadow: "0px 4px 12px rgba(0, 0, 0, 0.15)",
    }}
  >
    Estimate Selected ({selectedTasks.size})
  </Button>
</Box>
</>
    
  )
}
