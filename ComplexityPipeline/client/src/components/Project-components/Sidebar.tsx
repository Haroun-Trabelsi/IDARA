"use client"

import { Drawer, Box, Typography, List, ListItem, ListItemIcon, ListItemText, LinearProgress } from "@mui/material"
import React from "react"

interface SidebarItem {
  id: string | undefined
  progress: number
  color: string
}

interface SidebarProps {
  items: SidebarItem[]
}

export default function Sidebar({items}: SidebarProps) {
  return (
    <Drawer
      variant="permanent"
      sx={{
        width: 240,
        flexShrink: 0,
        "& .MuiDrawer-paper": {
          width: 240,
          boxSizing: "border-box",
          position: "relative",
          height: "100%",
          bgcolor: "#1a202c",
          borderRight: "1px solid #2d3748",
        },
      }}
    >
      <Box sx={{ p: 2 }}>

        <Typography variant="body2" color="#718096" sx={{ mb: 2, fontSize: "12px" }}>
          Lists
        </Typography>

        <List dense disablePadding>
          {items.map((item) => (
            <ListItem key={item.id} disablePadding sx={{ py: 0.5 }}>
              <ListItemIcon sx={{ minWidth: 24 }}>
                <Box
                  sx={{
                    width: 16,
                    height: 16,
                    bgcolor: "#2d3748",
                    borderRadius: 0.5,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <Box sx={{ width: 8, height: 8, bgcolor: "#4a5568", borderRadius: 0.5 }} />
                </Box>
              </ListItemIcon>
              <ListItemText
                primary={item.id}
                primaryTypographyProps={{
                  variant: "body2",
                  color: "#e2e8f0",
                  fontSize: "13px",
                }}
              />
              <Box sx={{ width: 64, ml: 1 }}>
                <LinearProgress
                  variant="determinate"
                  value={item.progress}
                  sx={{
                    height: 4,
                    borderRadius: 1,
                    bgcolor: "#2d3748",
                    "& .MuiLinearProgress-bar": {
                      bgcolor: item.color,
                    },
                  }}
                />
              </Box>
            </ListItem>
          ))}
        </List>
      </Box>
    </Drawer>
  )
}
