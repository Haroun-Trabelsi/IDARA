"use client";

import React from "react";
import { Box, Typography, Card, CardContent, Avatar, TextField, Button, Rating, FormControlLabel, Checkbox, Alert, IconButton } from "@mui/material";
import { Business, Feedback, Add, Delete } from "@mui/icons-material";

interface FeedbackPageProps {
  orgData: { name: string; organizationSize: number; id: string };
  rating: number | null;
  setRating: (value: number | null) => void;
  feedbackText: string;
  setFeedbackText: (text: string) => void;
  suggestFeatures: boolean;
  setSuggestFeatures: (value: boolean) => void;
  featureSuggestions: string[];
  handleAddFeatureSuggestion: () => void;
  handleRemoveFeatureSuggestion: (index: number) => void;
  handleFeatureSuggestionChange: (index: number, value: string) => void;
  handleSubmitFeedback: () => void;
  message: { type: "success" | "error"; text: string } | null;
}

export default function FeedbackPage({
  orgData,
  rating,
  setRating,
  feedbackText,
  setFeedbackText,
  suggestFeatures,
  setSuggestFeatures,
  featureSuggestions,
  handleAddFeatureSuggestion,
  handleRemoveFeatureSuggestion,
  handleFeatureSuggestionChange,
  handleSubmitFeedback,
  message,
}: FeedbackPageProps) {
  return (
    <Box>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 3 }}>
        <Typography variant="h4" sx={{ color: "#e2e8f0", fontWeight: 600 }}>
          We value your opinion
        </Typography>
        <Avatar sx={{ width: 40, height: 40, bgcolor: "transparent", border: "1px solid #2d3748" }}>
          <Feedback sx={{ fontSize: "20px", color: "#4299e1" }} />
        </Avatar>
      </Box>

      <Box sx={{ display: "flex", alignItems: "center", mb: 3 }}>
        <Avatar sx={{ width: 32, height: 32, bgcolor: "transparent", border: "1px solid #2d3748", mr: 2 }}>
          <Business sx={{ fontSize: "16px", color: "#4299e1" }} />
        </Avatar>
        <Typography variant="body1" sx={{ color: "#e2e8f0" }}>{orgData.name}</Typography>
      </Box>

      {message && <Alert severity={message.type} sx={{ mb: 3, borderRadius: "8px" }}>{message.text}</Alert>}

      {/* Carte unique contenant tous les éléments */}
      <Card sx={{ bgcolor: "#1a202c", border: "1px solid #2d3748", mb: 3, borderRadius: "12px" }}>
        <CardContent sx={{ p: 4 }}>
          {/* Section Rating */}
          <Box sx={{ mb: 4 }}>
            <Typography variant="h6" sx={{ color: "#e2e8f0", fontWeight: 500, fontSize: "16px", mb: 3 }}>
              Rate Your Experience
            </Typography>
            <Typography variant="body1" sx={{ color: "#e2e8f0", mb: 3, fontSize: "16px", textAlign: "center" }}>
              How would you rate your overall experience?
            </Typography>
            <Box sx={{ display: "flex", justifyContent: "center", mb: 2 }}>
              <Rating
                name="experience-rating"
                value={rating}
                onChange={(_, newValue) => setRating(newValue)}
                size="large"
                sx={{
                  fontSize: "2.5rem",
                  "& .MuiRating-iconFilled": { color: "#4299e1" },
                  "& .MuiRating-iconEmpty": { color: "#4a5568" },
                  "& .MuiRating-iconHover": { color: "#63b3ed" },
                }}
              />
            </Box>
          </Box>

          {/* Séparateur visuel */}
          <Box sx={{ borderBottom: "1px solid #2d3748", mb: 4 }} />

          {/* Section Feedback */}
          <Box>
            <Typography variant="h6" sx={{ color: "#e2e8f0", fontWeight: 500, fontSize: "16px", mb: 3 }}>
              Share Your Thoughts
            </Typography>
            
            <TextField
              label="Your Feedback"
              multiline
              rows={4}
              fullWidth
              value={feedbackText}
              onChange={(e) => setFeedbackText(e.target.value)}
              variant="outlined"
              sx={{
                mb: 3,
                "& .MuiOutlinedInput-root": {
                  bgcolor: "#2d3748",
                  color: "#e2e8f0",
                  "& fieldset": { borderColor: "#4a5568" },
                  "&:hover fieldset": { borderColor: "#718096" },
                },
                "& .MuiInputLabel-root": { color: "#a0aec0" },
              }}
            />

            <FormControlLabel
              control={
                <Checkbox
                  checked={suggestFeatures}
                  onChange={(e) => setSuggestFeatures(e.target.checked)}
                  sx={{
                    color: "#a0aec0",
                    "&.Mui-checked": { color: "#4299e1" },
                  }}
                />
              }
              label="Suggest new features"
              sx={{ color: "#e2e8f0", mb: 2 }}
            />

            {suggestFeatures && (
              <Box sx={{ mb: 3 }}>
                {featureSuggestions.map((suggestion, index) => (
                  <Box key={index} sx={{ display: "flex", alignItems: "center", mb: 2 }}>
                    <TextField
                      value={suggestion}
                      onChange={(e) => handleFeatureSuggestionChange(index, e.target.value)}
                      variant="outlined"
                      fullWidth
                      placeholder={`Feature suggestion ${index + 1}`}
                      sx={{
                        "& .MuiOutlinedInput-root": {
                          bgcolor: "#2d3748",
                          color: "#e2e8f0",
                          "& fieldset": { borderColor: "#4a5568" },
                          "&:hover fieldset": { borderColor: "#718096" },
                        },
                        "& .MuiInputLabel-root": { color: "#a0aec0" },
                      }}
                    />
                    {index > 0 && (
                      <IconButton
                        onClick={() => handleRemoveFeatureSuggestion(index)}
                        sx={{ color: "#ef4444", ml: 1 }}
                      >
                        <Delete />
                      </IconButton>
                    )}
                  </Box>
                ))}
                <Button
                  variant="outlined"
                  startIcon={<Add />}
                  onClick={handleAddFeatureSuggestion}
                  sx={{ 
                    color: "#4299e1", 
                    borderColor: "#4299e1", 
                    "&:hover": { 
                      borderColor: "#3182ce", 
                      bgcolor: "rgba(66, 153, 225, 0.1)" 
                    } 
                  }}
                >
                  Add Suggestion
                </Button>
              </Box>
            )}
          </Box>
        </CardContent>
      </Card>

      {/* Bouton Submit centré en dessous de la carte */}
      <Box sx={{ display: "flex", justifyContent: "center", mt: 3 }}>
        <Button
          variant="contained"
          onClick={handleSubmitFeedback}
          disabled={!rating || (suggestFeatures && !featureSuggestions.some((s) => s.trim()))}
          sx={{
            bgcolor: "#4299e1",
            "&:hover": { bgcolor: "#3182ce" },
            "&:disabled": { bgcolor: "#4a5568" },
            px: 4,
            py: 1.5,
            fontSize: "16px",
            fontWeight: 500,
          }}
        >
          Submit Feedback
        </Button>
      </Box>
    </Box>
  );
}