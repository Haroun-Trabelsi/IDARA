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
import { toFraction } from "../../utils/fractionUtils"; 
import React from "react"
import { useProject } from '../../contexts/ProjectContext';


  interface Difficulty {
  predicted_class: string;
  probabilities: {
    Easy: string;
    Medium: string;
    Hard: string;
  }
  eta?: number;

}

export interface Task {
  videos: any
  id: string;
  number: number;
  type: string; 
  status: "completed" | "in-progress" | "omitted" | "pending";
  assignee?: string;
  sequence?: string;
  description?: string;
  dueDate?: string;
  bidHours: number;
  taskName? : string;
  actualHours: number;
  level: number; 
  icon?: string;
  Difficulty?: Difficulty;
  notes_count : number;
}




interface TaskTableProps {
  project : String;
  tasks: Task[];
  setTasks: React.Dispatch<React.SetStateAction<Task[] | undefined>>;
}


export default function TaskTable({ project ,tasks, setTasks }: TaskTableProps) {

let headers: Record<string, string> = {};
    const token = localStorage.getItem('token');
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
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
async function fetchAllDifficulties(): Promise<Record<string, Record<string, any>>> {
  try {
    if (!selectedProject || !selectedProject.name) {
      console.error("No project selected.");
      return {};
    }
    const res = await fetch(`http://localhost:8080/results/results_by_task?project=${selectedProject.name}`,
      {headers}
    );
    if (!res.ok) throw new Error("Failed to fetch difficulty");
    const data = await res.json();
    return data || {};
  } catch (error) {
    console.error("Error fetching difficulties:", error);
    return {};
  }
}

async function loadDifficultiesForTasks(tasksToLoad: Task[]) {
  const allResultsResponse = await fetchAllDifficulties();
  const allResults = allResultsResponse.results || {};
  const updatedTasks = tasksToLoad.map((task) => {
    const sequenceKey = String(task.sequence); // make sure keys match backend type
    const shotKey = String(task.taskName); // match your backend's "descriptionId" (e.g. "100")

    if (
      sequenceKey &&
      shotKey &&
      allResults[sequenceKey] &&
      allResults[sequenceKey][shotKey]
    ) {
      const difficultyData = allResults[sequenceKey][shotKey];

      const Difficulty: Difficulty = {
        predicted_class: difficultyData.predicted_class || "",
        probabilities: difficultyData.probabilities || undefined,
        eta: Number(difficultyData.predicted_hours?.toFixed(2)) || 0,
      };

      return {
        ...task,
        Difficulty,
        EstimatedHours: Number(difficultyData.predicted_hours?.toFixed(2)) || 0,
      };
    }

    return {
      ...task,
      Difficulty: undefined,
      EstimatedHours: 0, // Default value if no difficulty data found
    };
  });

  setTasks(updatedTasks);
}




const { selectedProject } = useProject();
const [selectedTasks, setSelectedTasks] = useState<Set<string>>(new Set());
const [videoModalOpen, setVideoModalOpen] = useState(false);
const [videoOptions, setVideoOptions] = useState<any[]>([]);
const [activeTaskId, setActiveTaskId] = useState<string | null>(null);

useEffect(() => {
  if (!selectedProject?.name || tasks.length === 0) return;

  loadDifficultiesForTasks(tasks);
}, [selectedProject?.name, tasks.length]); // ✅ Only re-run when project name or task count changes


const formatDays = (hours: number) => {
  const days = Number((hours / 8).toFixed(1));
  return toFraction(days); // will show as "1/2", "5/8", etc.
};
const toggleTaskSelection = (taskId: string) => {
  setSelectedTasks(prev => {
    const updated = new Set(prev);
    updated.has(taskId) ? updated.delete(taskId) : updated.add(taskId);
    return updated;
  });
};
const handleOpenVideoSelector = async (taskId: string) => {
  setActiveTaskId(taskId);
  setVideoModalOpen(true);

  try {
    const res = await fetch(`http://localhost:8080/api/task/${taskId}/components`,{headers});
    const data = await res.json();

    const formattedVideos: any[] = data
      .map((v: any) => ({
        name: v.name,
        fileType: v.fileType,
        date: new Date(v.date),
        value: v.url.value,
      }))
      .reverse();

    setVideoOptions(formattedVideos);
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
const [videoUrl, setVideoUrl] = React.useState<string | null>(null);
const [open, setOpen] = React.useState(false);
const [loadingTaskId, setLoadingTaskId] = useState<string | null>(null);
const [queuedTaskIds, setQueuedTaskIds] = useState<Set<string>>(new Set());



const handleCloseVideo = () => {
  setOpen(false);
  setVideoUrl(null);
};

async function sendToFunction(selectedTaskObjects: Task[]): Promise<void> {
  // Initialize queue state with selected tasks IDs
  setQueuedTaskIds(new Set(selectedTaskObjects.map(task => task.id)));

  for (const task of selectedTaskObjects) {


    setQueuedTaskIds(prev => {
      const newQueued = new Set(prev);
      newQueued.delete(task.id);
      return newQueued;
    });

    setLoadingTaskId(task.id);

   const videoUrl = task.videos?.[0]?.value;
if (!videoUrl) {
  console.warn(`No video assigned to task ${task.id}`);
  setLoadingTaskId(null);
  continue;
}

try {
  const videoResponse = await fetch(videoUrl);
  if (!videoResponse.ok) throw new Error(`Failed to fetch video for task ${task.id}`);
  const blob = await videoResponse.blob();

  const sizeMB = blob.size / (1024 * 1024);
  console.log(`Video size : ${sizeMB.toFixed(2)} MB`);

  const formData = new FormData();
  formData.append("file", blob, `${task.id}.mp4`);

  formData.append(
    "original_filename",
    `Sequence - ${task.sequence} Task - ${task.taskName} Project Name - ${project}.mp4`
  );
  formData.append("description", task.description || "");
  formData.append("notes_count", task.notes_count?.toString() || "0");

  const uploadResponse = await fetch("http://localhost:8089/upload_video", {
    method: "POST",
    body: formData,
  });


      if (!uploadResponse.ok) throw new Error(`Upload failed with status ${uploadResponse.status}`);
      const { job_id } = await uploadResponse.json();

      const pollUrl = `http://localhost:8089/result/${job_id}`;
      const baseInterval = 2000; // ms
      const intervalPerMB = 200; // ms extra per MB
      const pollInterval = baseInterval + sizeMB * intervalPerMB;

      const baseAttempts = 30;
      const extraAttemptsPerMB = 5;
      const maxAttempts = Math.ceil(baseAttempts + sizeMB * extraAttemptsPerMB);

      let resultReceived = false;

      for (let attempt = 0; attempt < maxAttempts; attempt++) {
        // Skip first 5 polling attempts with wait only
        if (attempt < 5) {
          await new Promise(resolve => setTimeout(resolve, pollInterval));
          continue;
        }

        const res = await fetch(pollUrl);
        if (!res.ok) {
          console.error(`Failed to fetch result for job ${job_id}, status: ${res.status}`);
          break;
        }
        const data = await res.json();

        if (data.status === "done" && data.result) {
        resultReceived = true;

        const difficultyInfo = data.result.classification_result;
        const predictedHours = data.result.predicted_vfx_hours ?? 0;

        setTasks(prev =>
          (prev || []).map(t =>
            t.id === task.id ? { 
              ...t, 
              Difficulty: {
                ...difficultyInfo,
                eta: predictedHours,
              },
              eta: predictedHours
            } : t
          )
        );
        break;
      }


        await new Promise(resolve => setTimeout(resolve, pollInterval));
      }

      if (!resultReceived) {
        alert(`Task ${task.id} timed out after ${maxAttempts} attempts.`);
      }
    } catch (err) {
      console.error(`Error processing task ${task.id}:`, err);
      alert(`Failed to process video for task ${task.id}.`);
    } finally {
      setLoadingTaskId(null);
    }
  }

  setSelectedTasks(new Set());
  setQueuedTaskIds(new Set());
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
                width: 150,
              }}
            >
              Tasks
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
              Type
            </TableCell>
            <TableCell
              sx={{
                width: 160,
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
                width: 120,
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
                width: 200,
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
                width: 160,
                bgcolor: "#1a202c",
                color: "#718096",
                borderBottom: "1px solid #2d3748",
                fontSize: "12px",
                fontWeight: 500,
              }}
            >
            Estimated Bid Hours
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
              Bid Hours
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
              Bid Days
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
              +/- Days
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
    <Box
      onClick={() => handleOpenVideoSelector(task.id)}
      sx={{ ml: 1, cursor: "pointer", "&:hover": { textDecoration: "underline" } }}
    >
      <Typography
  variant="body2"
  sx={{
    color: task.videos?.length > 0 ? "#16a34a" : "#ff0000ff", // green if selected, yellow if not
    fontSize: "13px",
    fontWeight: 500,
  }}
>
  {task.videos?.length > 0 ? "Video Selected!" : "Select Video"}
</Typography>

          <Typography variant="body2" sx={{ color: "#e2e8f0", fontSize: "13px" }}>
  {task.sequence || ""} / {task.taskName || ""}
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
                    {task.description}
                </Typography>
              </TableCell>

              <TableCell sx={{ p: 1, borderBottom: "1px solid #2d3748" }}>
                <Typography variant="body2" sx={{ color: "#a0aec0", fontSize: "13px" }}>
                  {task.dueDate || ""}
                </Typography>
              </TableCell>
<TableCell align="right" sx={{ p: 1, borderBottom: "1px solid #2d3748" }}>
  <Typography variant="body2" sx={{ color: "#a0aec0", fontSize: "13px" }}>
    {task.Difficulty?.eta ? task.Difficulty?.eta : "N/A"}
  </Typography>
</TableCell>
<TableCell align="right" sx={{ p: 1, borderBottom: "1px solid #2d3748" }}>
  <Typography variant="body2" sx={{ color: "#a0aec0", fontSize: "13px" }}>
    { task.bidHours.toFixed(2) } 
  </Typography>
</TableCell>
 <TableCell align="right" sx={{ p: 1, borderBottom: "1px solid #2d3748" }}>
  <Typography variant="body2" sx={{ color: "#a0aec0", fontSize: "13px" }}>
    { formatDays(task.bidHours)} 
  </Typography>
</TableCell>

<TableCell align="right" sx={{ p: 1, borderBottom: "1px solid #2d3748" }}>
  <Typography variant="body2" sx={{ color: "#a0aec0", fontSize: "13px" }}>
   { formatDays(task.actualHours)}
  </Typography>
</TableCell>

<TableCell>
  {loadingTaskId === task.id ? (
    <Box sx={{ display: 'flex', alignItems: 'center' }}>
      <CircularProgress size={20} sx={{ mr: 1 }} />
      <Typography variant="body2" color="textSecondary">Processing...</Typography>
      
    </Box>
  ) : queuedTaskIds.has(task.id) ? (
    <Box display="flex" alignItems="center" flexWrap="wrap" gap={1}>
  <Typography variant="body2" color="textSecondary">
    Queued 
  </Typography>

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
    display: "flex",
    gap: 2, // spacing between buttons
  }}
>
  <Button
    variant="outlined"
    color="secondary"
    size="medium"
    disabled={selectedTasks.size === 0}
    onClick={() => setSelectedTasks(new Set())}
    sx={{
      px: 4,
      py: 1.5,
      fontSize: "14px",
      borderRadius: 2,
      boxShadow: "0px 4px 12px rgba(0, 0, 0, 0.10)",
    }}
  >
    Clear Selection
  </Button>

  <Button
    variant="contained"
    color="primary"
    size="medium"
    disabled={selectedTasks.size === 0}
    onClick={() => {
      const selectedTaskObjects = tasks.filter(task => selectedTasks.has(task.id));
      sendToFunction(selectedTaskObjects);
    }}
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
          onClick={() => handleVideoSelect(video.value)}
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
