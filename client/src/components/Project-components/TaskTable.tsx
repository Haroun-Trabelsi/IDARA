"use client"

import { useEffect, useState } from "react"
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
  Dialog,
  DialogContent,
  CircularProgress,
} from "@mui/material"
import {
  PlayCircleOutline,
} from "@mui/icons-material"
import React from "react"
import { useProject } from '../../contexts/ProjectContext';


  interface Difficulty {
  predicted_class: string;
  probabilities: {
    Easy: string;
    Medium: string;
    Hard: string;
  }
}

export interface Task {
  videos: any
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
  Difficulty?: Difficulty;
}




interface TaskTableProps {
  project : String;
  tasks: Task[];
  setTasks: React.Dispatch<React.SetStateAction<Task[] | undefined>>;
}


export default function TaskTable({ project ,tasks, setTasks }: TaskTableProps) {


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
async function fetchDifficulty(taskId: string): Promise<Difficulty | null> {
  try {
    if (!selectedProject || !selectedProject.name) {
      console.error("No project selected.");
      return null;
    }

    const res = await fetch(`http://localhost:8080/results/results_by_task?project=${selectedProject.name}`);
    if (!res.ok) throw new Error("Failed to fetch difficulty");

    const data = await res.json();
    const allResults = data.results;

    for (const taskMap of Object.values(allResults)) {
      if (taskMap && typeof taskMap === "object") {
        const raw = (taskMap as Record<string, any>)[taskId];
        if (
          raw &&
          typeof raw.predicted_class === "string" &&
          raw.probabilities &&
          typeof raw.probabilities.Easy === "string" &&
          typeof raw.probabilities.Medium === "string" &&
          typeof raw.probabilities.Hard === "string"
        ) {
          return raw as Difficulty;
        }
      }
    }

    return null;
  } catch (error) {
    console.error(`Error fetching difficulty for task ${taskId}:`, error);
    return null;
  }
}

async function loadDifficultiesForTasks(tasksToLoad: Task[]) {
  const updatedTasks = await Promise.all(
    tasksToLoad.map(async (task) => {
      const difficulty = await fetchDifficulty(task.id);
      return difficulty ? { ...task, Difficulty: difficulty } : task;
    })
  );
  setTasks(updatedTasks);
}
const [hasLoadedDifficulties, setHasLoadedDifficulties] = useState(false);

useEffect(() => {
  if (!hasLoadedDifficulties && tasks.length > 0) {
    loadDifficultiesForTasks(tasks);
    setHasLoadedDifficulties(true);
    console.log("Difficulties loaded for tasks:", tasks);
  }
}, [tasks, hasLoadedDifficulties]);

const { selectedProject } = useProject();
const [selectedTasks, setSelectedTasks] = useState<Set<string>>(new Set());
const [videoModalOpen, setVideoModalOpen] = useState(false);
const [videoOptions, setVideoOptions] = useState<any[]>([]);
const [activeTaskId, setActiveTaskId] = useState<string | null>(null);

const toggleTaskSelection = (taskId: string) => {
  setSelectedTasks(prev => {
    const updated = new Set(prev);
    updated.has(taskId) ? updated.delete(taskId) : updated.add(taskId);
    return updated;
  });
};
const handleOpenVideoSelector = async (taskId: string) => {
  const task = tasks.find(t => t.id === taskId);
  if (!task) return;

  try {
    const res = await fetch(`http://localhost:8080/api/task/${taskId}/components`);
    const videoList = await res.json();

    // Attach videos with name and date
    const options = videoList.map((v: any) => ({
      name: v.name,
      fileType: v.fileType,
      date: new Date(v.date).toLocaleDateString(),
      value: v.url.value
    }));

    setVideoOptions(options);
    setActiveTaskId(taskId);
    setVideoModalOpen(true);
  } catch (err) {
    console.error("Failed to load videos", err);
  }
};


const handleVideoSelect = (url: string) => {
  setTasks((prev: Task[] | undefined) =>
    (prev || []).map((task: Task) =>
      task.id === activeTaskId ? { ...task, videos: [{ value: url }] } : task
    )
  );
  setVideoModalOpen(false);
  setVideoOptions([]);
  setActiveTaskId(null);
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
const [videoUrl, setVideoUrl] = React.useState<string | null>(null);
const [open, setOpen] = React.useState(false);
const [loadingTaskId, setLoadingTaskId] = useState<string | null>(null);
const [taskJobIds, setTaskJobIds] = useState<Record<string, string>>({});

const cancelJob = async (jobId: string) => {
  try {
    const res = await fetch(`http://localhost:8089/cancel/${jobId}`, {
      method: "POST",
    });
    const data = await res.json();
    if (data.status === "cancelled") {
      alert("Processing cancelled.");
      // Optional: update UI to reflect cancellation
    } else {
      alert("Could not cancel: " + data.message);
    }
  } catch (err) {
    console.error("Cancellation failed:", err);
  }
};


const handleOpenVideo = (url: string) => {
  setVideoUrl(url);
  setOpen(true);
};
const handleCloseVideo = () => {
  setOpen(false);
  setVideoUrl(null);
};

async function sendToFunction(selectedTaskObjects: Task[]): Promise<void> {
  for (const task of selectedTaskObjects) {
    const videoUrl = task.videos?.[0]?.value;

    if (!videoUrl) {
      alert(`No video assigned to task ${task.id}. Please select a video before sending.`);
      continue;
    }

    try {
      setLoadingTaskId(task.id); // UI indicator

      // Delay first request by 3 seconds
      await new Promise(resolve => setTimeout(resolve, 3000));

      // Step 1: Download video
      const videoResponse = await fetch(videoUrl.value);
      console.log(videoUrl.value);

      if (!videoResponse.ok) throw new Error(`Failed to fetch video for task ${task.id}`);
      const blob = await videoResponse.blob();

      const formData = new FormData();
      formData.append("file", blob, `${task.id}.mp4`);
      formData.append("original_filename", `Sequence - ${task.sequence} Task - ${task.description} Project Name - ${project}.mp4`);

      // Step 2: Upload to server
      const uploadResponse = await fetch("http://localhost:8089/upload_video", {
        method: "POST",
        body: formData,
      });

      if (!uploadResponse.ok) {
        throw new Error(`Upload failed with status ${uploadResponse.status}`);
      }

      const { job_id } = await uploadResponse.json();
      console.log(`Video uploaded for task ${task.id}, job ID: ${job_id}`);
      setTaskJobIds(prev => ({ ...prev, [task.id]: job_id }));

      // Step 3: Poll for result
      const pollUrl = `http://localhost:8089/result/${job_id}`;
      const maxAttempts = 30;
      let resultReceived = false;

      for (let attempt = 0; attempt < maxAttempts; attempt++) {
        const res = await fetch(pollUrl);
        const data = await res.json();

        if (data.status === "done" && data.result) {
          console.log(`Task ${task.id} completed:`, data.result);
          resultReceived = true;
          break;
        } else if (data.status === "processing") {
          console.log(`Task ${task.id} is still processing (attempt ${attempt + 1})`);
        } else {
          console.error(`Unexpected status or response for task ${task.id}:`, data);
          break;
        }

        await new Promise(resolve => setTimeout(resolve, 1000 * Math.min(attempt + 1, 5))); // backoff
      }

      if (!resultReceived) {
        alert(`Task ${task.id} timed out. Please try again later.`);
      }

    } catch (err) {
      console.error(`Error processing task ${task.id}:`, err);
      alert(`Failed to process video for task ${task.id}.`);
    } finally {
      setLoadingTaskId(null);
    }
  }
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
            <TableCell
              align="center"
              sx={{
                width: 300,
                bgcolor: "#1a202c",
                color: "#718096",
                borderBottom: "1px solid #2d3748",
                fontSize: "12px",
                fontWeight: 500,
              }}
            >
              Difficulty
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
                  
                    {task.videos && task.videos.length > 0 ? (
    <IconButton
      size="small"
      onClick={() => handleOpenVideo(task.videos[0].value)}
      sx={{ color: "#63b3ed", p: 0.5 }}
      aria-label="play video"
    >
      <PlayCircleOutline fontSize="small" />
    </IconButton>
  ) : (
    <Typography variant="body2" sx={{ color: "#a0aec0", fontSize: "13px" }}>
      No video
    </Typography>
  )}
    <Box
      onClick={() => handleOpenVideoSelector(task.id)}
      sx={{ ml: 1, cursor: "pointer", "&:hover": { textDecoration: "underline" } }}
    >
      <Typography variant="body2" sx={{ color: "#e2e8f0", fontSize: "13px" }}>
        {task.sequence || ""} / {task.description || ""} / {task.icon || ''}
      </Typography>
    </Box>


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
<TableCell>
  {loadingTaskId === task.id ? (
    <Box sx={{ display: 'flex', alignItems: 'center' }}>
      <CircularProgress size={20} sx={{ mr: 1 }} />
      <Typography variant="body2" color="textSecondary">Processing...</Typography>
      <Button
        color="error"
        variant="outlined"
        onClick={() => cancelJob(taskJobIds[task.id])}
        disabled={!taskJobIds[task.id]}
      >
        Cancel
      </Button>
    </Box>
  ) : task.Difficulty ? (
    <Box>
      <Typography variant="body2" fontWeight="bold">
        {task.Difficulty.predicted_class}
      </Typography>
      <Typography variant="caption" color="textSecondary">
        Easy: {task.Difficulty.probabilities?.Easy || "N/A"}, 
        Medium: {task.Difficulty.probabilities?.Medium || "N/A"}, 
        Hard: {task.Difficulty.probabilities?.Hard || "N/A"}
      </Typography>
    </Box>
  ) : (
    <Typography variant="body2" color="textSecondary">Not processed</Typography>
  )}
</TableCell>


            </TableRow>
          ))}
        </TableBody>
      </Table>
<Dialog open={open} onClose={handleCloseVideo} maxWidth="md" fullWidth>
  <DialogContent sx={{ position: "relative", p: 0 }}>
    <IconButton
      onClick={handleCloseVideo}
      sx={{ position: "absolute", top: 8, right: 8, color: "white", zIndex: 1 }}
      aria-label="close"
    >
    </IconButton>
    <video
      src={videoUrl || ""}
      controls
      autoPlay
      style={{ width: "100%", height: "auto", backgroundColor: "black" }}
    />
  </DialogContent>
</Dialog>

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
<Dialog open={videoModalOpen} onClose={() => setVideoModalOpen(false)} maxWidth="md" fullWidth>
  <DialogContent>
    <Typography variant="h6" sx={{ mb: 2, color: "#e2e8f0" }}>
      Select a version for this task
    </Typography>

    {videoOptions
  .filter((video) => video.fileType === ".mp4" || video.fileType === ".mov")
  .slice(-4)
  .map((video, index) => (
    <Box key={index}>
      <video
        src={video.value}
        controls
        style={{
          width: "100%",
          maxHeight: "300px",
          borderRadius: "4px",
          backgroundColor: "black",
        }}
      />

      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          mt: 1,
          color: "#cbd5e0",
          fontSize: "13px",
        }}
      >
        <span>
          <b>Name:</b> {video.name}
        </span>
        <span>
          <b>Type:</b> {video.fileType}
        </span>
        <span>
          <b>Date:</b> {new Date(video.date).toLocaleDateString()}
        </span>
      </Box>

      <Box sx={{ display: "flex", justifyContent: "flex-end", mt: 1 }}>
        <Button
          size="small"
          variant="outlined"
          onClick={() => handleVideoSelect(video)}
        >
          Select
        </Button>
      </Box>
    </Box>
  ))}

  </DialogContent>
</Dialog>

</>
    
  )
}
