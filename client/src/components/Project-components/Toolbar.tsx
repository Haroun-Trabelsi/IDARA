"use client"

import { useState } from "react"
import { Box, Button, IconButton, Menu, MenuItem } from "@mui/material"
import {
  Add,
  FileDownload,
  FileUpload,
  MoreHoriz,
  ViewList,
  ViewModule,
  CalendarToday,
  Visibility,
  Refresh,
  Description,
  FilterList,
  Settings,
  KeyboardArrowDown,
} from "@mui/icons-material"
import React from "react"

export default function Toolbar() {
  const [createAnchorEl, setCreateAnchorEl] = useState<null | HTMLElement>(null)
  const [exportAnchorEl, setExportAnchorEl] = useState<null | HTMLElement>(null)
  const [viewAnchorEl, setViewAnchorEl] = useState<null | HTMLElement>(null)

  return (
    <Box sx={{ borderBottom: "1px solid #2d3748", bgcolor: "#1a202c" }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", px: 2, py: 1.5 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <Button
            variant="contained"
            startIcon={<Add sx={{ fontSize: "16px" }} />}
            size="small"
            onClick={(e) => setCreateAnchorEl(e.currentTarget)}
            sx={{
              bgcolor: "#4299e1",
              "&:hover": {
                bgcolor: "#3182ce",
              },
              textTransform: "none",
              fontSize: "13px",
            }}
          >
            Create
          </Button>
          <Menu
            anchorEl={createAnchorEl}
            open={Boolean(createAnchorEl)}
            onClose={() => setCreateAnchorEl(null)}
            PaperProps={{
              sx: {
                bgcolor: "#2d3748",
                border: "1px solid #4a5568",
              },
            }}
          >
            <MenuItem sx={{ color: "#e2e8f0" }}>New Task</MenuItem>
            <MenuItem sx={{ color: "#e2e8f0" }}>New Project</MenuItem>
          </Menu>

          <Button
            startIcon={<FileUpload sx={{ fontSize: "16px" }} />}
            size="small"
            sx={{
              color: "#a0aec0",
              textTransform: "none",
              fontSize: "13px",
              "&:hover": {
                bgcolor: "rgba(255, 255, 255, 0.05)",
                color: "#e2e8f0",
              },
            }}
          >
            Import
          </Button>

          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, ml: 1 }}>
            <IconButton size="small" sx={{ color: "#a0aec0" }}>
              <MoreHoriz fontSize="small" />
            </IconButton>
            <IconButton size="small" sx={{ color: "#a0aec0" }}>
              <Refresh fontSize="small" />
            </IconButton>
            <IconButton size="small" sx={{ color: "#a0aec0" }}>
              <Description fontSize="small" />
            </IconButton>
          </Box>

          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, ml: 2 }}>
            <IconButton size="small" sx={{ color: "#a0aec0" }}>
              <ViewList fontSize="small" />
            </IconButton>
            <IconButton size="small" sx={{ color: "#a0aec0" }}>
              <ViewModule fontSize="small" />
            </IconButton>
            <IconButton size="small" sx={{ color: "#a0aec0" }}>
              <CalendarToday fontSize="small" />
            </IconButton>
          </Box>
        </Box>

        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <Button
            size="small"
            sx={{
              color: "#a0aec0",
              textTransform: "none",
              fontSize: "13px",
              "&:hover": {
                bgcolor: "rgba(255, 255, 255, 0.05)",
                color: "#e2e8f0",
              },
            }}
          >
            Group
          </Button>
          <Button
            startIcon={<Visibility sx={{ fontSize: "16px" }} />}
            size="small"
            onClick={(e) => setViewAnchorEl(e.currentTarget)}
            sx={{
              color: "#a0aec0",
              textTransform: "none",
              fontSize: "13px",
              "&:hover": {
                bgcolor: "rgba(255, 255, 255, 0.05)",
                color: "#e2e8f0",
              },
            }}
          >
            View
          </Button>
          <Menu
            anchorEl={viewAnchorEl}
            open={Boolean(viewAnchorEl)}
            onClose={() => setViewAnchorEl(null)}
            PaperProps={{
              sx: {
                bgcolor: "#2d3748",
                border: "1px solid #4a5568",
              },
            }}
          >
            <MenuItem sx={{ color: "#e2e8f0" }}>Default View</MenuItem>
            <MenuItem sx={{ color: "#e2e8f0" }}>Custom View</MenuItem>
          </Menu>
          <Button
            endIcon={<KeyboardArrowDown sx={{ fontSize: "16px" }} />}
            size="small"
            sx={{
              color: "#a0aec0",
              textTransform: "none",
              fontSize: "13px",
              "&:hover": {
                bgcolor: "rgba(255, 255, 255, 0.05)",
                color: "#e2e8f0",
              },
            }}
          >
            Default
          </Button>
          <Button
            startIcon={<FileDownload sx={{ fontSize: "16px" }} />}
            endIcon={<KeyboardArrowDown sx={{ fontSize: "16px" }} />}
            size="small"
            onClick={(e) => setExportAnchorEl(e.currentTarget)}
            sx={{
              color: "#a0aec0",
              textTransform: "none",
              fontSize: "13px",
              "&:hover": {
                bgcolor: "rgba(255, 255, 255, 0.05)",
                color: "#e2e8f0",
              },
            }}
          >
            Export
          </Button>
          <Menu
            anchorEl={exportAnchorEl}
            open={Boolean(exportAnchorEl)}
            onClose={() => setExportAnchorEl(null)}
            PaperProps={{
              sx: {
                bgcolor: "#2d3748",
                border: "1px solid #4a5568",
              },
            }}
          >
            <MenuItem sx={{ color: "#e2e8f0" }}>Export CSV</MenuItem>
            <MenuItem sx={{ color: "#e2e8f0" }}>Export PDF</MenuItem>
          </Menu>
          <IconButton size="small" sx={{ color: "#a0aec0" }}>
            <Refresh fontSize="small" />
          </IconButton>
          <IconButton size="small" sx={{ color: "#a0aec0" }}>
            <Settings fontSize="small" />
          </IconButton>
          <Button
            startIcon={<FilterList sx={{ fontSize: "16px" }} />}
            size="small"
            sx={{
              color: "#a0aec0",
              textTransform: "none",
              fontSize: "13px",
              "&:hover": {
                bgcolor: "rgba(255, 255, 255, 0.05)",
                color: "#e2e8f0",
              },
            }}
          >
            Filters
          </Button>
        </Box>
      </Box>
    </Box>
  )
}
