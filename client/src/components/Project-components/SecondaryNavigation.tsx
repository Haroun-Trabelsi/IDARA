"use client"

import { Box, Button, Typography } from "@mui/material"
import { KeyboardArrowDown } from "@mui/icons-material"
import React from "react"

export default function SecondaryNavigation() {
  return (
    <Box sx={{ borderBottom: "1px solid #2d3748", bgcolor: "#1a202c" }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", px: 2, py: 1.5 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 3 }}>
          <Button
            sx={{
              color: "#4299e1",
              textTransform: "none",
              fontSize: "14px",
              fontWeight: 500,
              "&:hover": {
                bgcolor: "rgba(66, 153, 225, 0.1)",
              },
            }}
          >
            Tasks
          </Button>
          <Button
            sx={{
              color: "#a0aec0",
              textTransform: "none",
              fontSize: "14px",
              "&:hover": {
                bgcolor: "rgba(255, 255, 255, 0.05)",
                color: "#e2e8f0",
              },
            }}
          >
            Versions
          </Button>
          <Button
            endIcon={<KeyboardArrowDown sx={{ fontSize: "16px" }} />}
            sx={{
              color: "#a0aec0",
              textTransform: "none",
              fontSize: "14px",
              "&:hover": {
                bgcolor: "rgba(255, 255, 255, 0.05)",
                color: "#e2e8f0",
              },
            }}
          >
            Team planner
          </Button>
          <Button
            endIcon={<KeyboardArrowDown sx={{ fontSize: "16px" }} />}
            sx={{
              color: "#a0aec0",
              textTransform: "none",
              fontSize: "14px",
              "&:hover": {
                bgcolor: "rgba(255, 255, 255, 0.05)",
                color: "#e2e8f0",
              },
            }}
          >
            Dashboards
          </Button>
        </Box>
        <Typography variant="body2" color="#718096" sx={{ fontSize: "13px" }}>
          7760
        </Typography>
      </Box>
    </Box>
  )
}
