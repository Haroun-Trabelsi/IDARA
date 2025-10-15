/**
 * Mock Data for IDara VFX Project Management Platform
 * This file contains dummy static data to showcase the application's functionality
 * when API and database access is unavailable.
 */

export const mockAccounts = [
  {
    _id: "507f1f77bcf86cd799439011",
    name: "John",
    surname: "Doe",
    email: "john.doe@vfxstudio.com",
    role: "admin" as const,
    isVerified: true,
    mfaEnabled: false,
    organizationName: "VFX Studio Alpha",
    status: "AdministratorOrganization" as const,
    canInvite: true,
    mustCompleteProfile: false,
    teamSize: "10-50",
    region: "North America",
    ID_Organization: "ORG-001",
    lastConnexion: new Date("2025-10-15T10:30:00Z"),
    createdAt: new Date("2025-01-15T08:00:00Z"),
    updatedAt: new Date("2025-10-15T10:30:00Z"),
  },
  {
    _id: "507f1f77bcf86cd799439012",
    name: "Sarah",
    surname: "Johnson",
    email: "sarah.johnson@vfxstudio.com",
    role: "user" as const,
    isVerified: true,
    mfaEnabled: true,
    organizationName: "VFX Studio Alpha",
    status: "accepted" as const,
    invitedBy: "507f1f77bcf86cd799439011",
    canInvite: true,
    mustCompleteProfile: false,
    teamSize: "10-50",
    region: "North America",
    lastConnexion: new Date("2025-10-14T16:45:00Z"),
    createdAt: new Date("2025-02-01T09:30:00Z"),
    updatedAt: new Date("2025-10-14T16:45:00Z"),
  },
  {
    _id: "507f1f77bcf86cd799439013",
    name: "Mike",
    surname: "Chen",
    email: "mike.chen@creativevfx.com",
    role: "admin" as const,
    isVerified: true,
    mfaEnabled: false,
    organizationName: "Creative VFX House",
    status: "AdministratorOrganization" as const,
    canInvite: true,
    mustCompleteProfile: false,
    teamSize: "50-200",
    region: "Asia Pacific",
    ID_Organization: "ORG-002",
    lastConnexion: new Date("2025-10-15T03:20:00Z"),
    createdAt: new Date("2025-03-10T12:00:00Z"),
    updatedAt: new Date("2025-10-15T03:20:00Z"),
  },
  {
    _id: "507f1f77bcf86cd799439014",
    name: "Emma",
    surname: "Williams",
    email: "emma.williams@vfxstudio.com",
    role: "user" as const,
    isVerified: true,
    mfaEnabled: false,
    organizationName: "VFX Studio Alpha",
    status: "accepted" as const,
    invitedBy: "507f1f77bcf86cd799439011",
    canInvite: false,
    mustCompleteProfile: false,
    teamSize: "10-50",
    region: "North America",
    lastConnexion: new Date("2025-10-13T14:20:00Z"),
    createdAt: new Date("2025-04-20T11:15:00Z"),
    updatedAt: new Date("2025-10-13T14:20:00Z"),
  },
  {
    _id: "507f1f77bcf86cd799439015",
    name: "Alex",
    surname: "Rodriguez",
    email: "alex.rodriguez@vfxstudio.com",
    role: "user" as const,
    isVerified: false,
    mfaEnabled: false,
    organizationName: "VFX Studio Alpha",
    status: "pending" as const,
    invitedBy: "507f1f77bcf86cd799439011",
    canInvite: false,
    mustCompleteProfile: true,
    teamSize: "10-50",
    region: "North America",
    invitedDate: new Date("2025-10-10T09:00:00Z"),
    createdAt: new Date("2025-10-10T09:00:00Z"),
    updatedAt: new Date("2025-10-10T09:00:00Z"),
  },
];

export const mockProjects = [
  {
    _id: "proj-001",
    name: "Galactic Warriors",
    organizationId: "ORG-001",
    description: "Sci-fi action movie with extensive VFX work",
    status: "active",
    createdAt: new Date("2025-01-20T00:00:00Z"),
    totalShots: 145,
    completedShots: 67,
    sequences: ["seq-001", "seq-002", "seq-003"],
  },
  {
    _id: "proj-002",
    name: "Ocean's Mystery",
    organizationId: "ORG-001",
    description: "Underwater fantasy adventure",
    status: "active",
    createdAt: new Date("2025-03-15T00:00:00Z"),
    totalShots: 89,
    completedShots: 23,
    sequences: ["seq-004", "seq-005"],
  },
  {
    _id: "proj-003",
    name: "City of Tomorrow",
    organizationId: "ORG-002",
    description: "Futuristic cityscape project",
    status: "planning",
    createdAt: new Date("2025-05-01T00:00:00Z"),
    totalShots: 210,
    completedShots: 0,
    sequences: ["seq-006", "seq-007", "seq-008", "seq-009"],
  },
];

export const mockSequences = [
  { _id: "seq-001", projectId: "proj-001", name: "SQ010", description: "Opening space battle" },
  { _id: "seq-002", projectId: "proj-001", name: "SQ020", description: "Alien planet landing" },
  { _id: "seq-003", projectId: "proj-001", name: "SQ030", description: "Final confrontation" },
  { _id: "seq-004", projectId: "proj-002", name: "SQ010", description: "Underwater chase" },
  { _id: "seq-005", projectId: "proj-002", name: "SQ020", description: "Deep sea discovery" },
  { _id: "seq-006", projectId: "proj-003", name: "SQ010", description: "City flyover" },
  { _id: "seq-007", projectId: "proj-003", name: "SQ020", description: "Street level action" },
  { _id: "seq-008", projectId: "proj-003", name: "SQ030", description: "Building collapse" },
  { _id: "seq-009", projectId: "proj-003", name: "SQ040", description: "Rooftop finale" },
];

export const mockResults = [
  {
    _id: "result-001",
    filename: "GW_SQ010_SH0010_v001.mov",
    project: "Galactic Warriors",
    sequence: "SQ010",
    shot: "SH0010",
    classification_result: {
      predicted_class: "Hard",
      confidence: "0.87",
      probabilities: {
        Easy: "0.05",
        Medium: "0.08",
        Hard: "0.87",
      },
    },
    complexity_scores: {
      blur_score: 0.72,
      focus_pull_score: 0.85,
      noise_score: 0.45,
      camera_motion_score: 0.91,
      parallax_score: 0.78,
      overlap_score: 0.82,
      distortion_score: 0.56,
      lighting_score: 0.68,
      zoom_score: 0.73,
    },
    predicted_vfx_hours: 48.5,
    processing_time_seconds: 12.4,
    version: 1,
    createdAt: new Date("2025-08-15T14:23:00Z"),
  },
  {
    _id: "result-002",
    filename: "GW_SQ010_SH0020_v001.mov",
    project: "Galactic Warriors",
    sequence: "SQ010",
    shot: "SH0020",
    classification_result: {
      predicted_class: "Medium",
      confidence: "0.76",
      probabilities: {
        Easy: "0.12",
        Medium: "0.76",
        Hard: "0.12",
      },
    },
    complexity_scores: {
      blur_score: 0.35,
      focus_pull_score: 0.42,
      noise_score: 0.28,
      camera_motion_score: 0.55,
      parallax_score: 0.48,
      overlap_score: 0.52,
      distortion_score: 0.31,
      lighting_score: 0.44,
      zoom_score: 0.38,
    },
    predicted_vfx_hours: 24.2,
    processing_time_seconds: 9.8,
    version: 1,
    createdAt: new Date("2025-08-16T10:15:00Z"),
  },
  {
    _id: "result-003",
    filename: "GW_SQ020_SH0010_v002.mov",
    project: "Galactic Warriors",
    sequence: "SQ020",
    shot: "SH0010",
    classification_result: {
      predicted_class: "Hard",
      confidence: "0.92",
      probabilities: {
        Easy: "0.02",
        Medium: "0.06",
        Hard: "0.92",
      },
    },
    complexity_scores: {
      blur_score: 0.88,
      focus_pull_score: 0.91,
      noise_score: 0.67,
      camera_motion_score: 0.94,
      parallax_score: 0.89,
      overlap_score: 0.93,
      distortion_score: 0.72,
      lighting_score: 0.81,
      zoom_score: 0.86,
    },
    predicted_vfx_hours: 67.8,
    processing_time_seconds: 15.2,
    version: 2,
    createdAt: new Date("2025-09-02T11:45:00Z"),
  },
  {
    _id: "result-004",
    filename: "OM_SQ010_SH0005_v001.mov",
    project: "Ocean's Mystery",
    sequence: "SQ010",
    shot: "SH0005",
    classification_result: {
      predicted_class: "Easy",
      confidence: "0.82",
      probabilities: {
        Easy: "0.82",
        Medium: "0.14",
        Hard: "0.04",
      },
    },
    complexity_scores: {
      blur_score: 0.15,
      focus_pull_score: 0.22,
      noise_score: 0.18,
      camera_motion_score: 0.25,
      parallax_score: 0.21,
      overlap_score: 0.28,
      distortion_score: 0.19,
      lighting_score: 0.32,
      zoom_score: 0.16,
    },
    predicted_vfx_hours: 8.5,
    processing_time_seconds: 7.3,
    version: 1,
    createdAt: new Date("2025-09-20T09:30:00Z"),
  },
  {
    _id: "result-005",
    filename: "OM_SQ020_SH0012_v001.mov",
    project: "Ocean's Mystery",
    sequence: "SQ020",
    shot: "SH0012",
    classification_result: {
      predicted_class: "Medium",
      confidence: "0.71",
      probabilities: {
        Easy: "0.18",
        Medium: "0.71",
        Hard: "0.11",
      },
    },
    complexity_scores: {
      blur_score: 0.48,
      focus_pull_score: 0.53,
      noise_score: 0.41,
      camera_motion_score: 0.59,
      parallax_score: 0.51,
      overlap_score: 0.56,
      distortion_score: 0.44,
      lighting_score: 0.49,
      zoom_score: 0.47,
    },
    predicted_vfx_hours: 28.3,
    processing_time_seconds: 10.5,
    version: 1,
    createdAt: new Date("2025-10-01T15:20:00Z"),
  },
];

export const mockContactMessages = [
  {
    _id: "msg-001",
    name: "David Thompson",
    email: "david.thompson@email.com",
    subject: "Question about enterprise pricing",
    message: "Hi, I'm interested in learning more about your enterprise plans for larger VFX studios. Could you provide more information?",
    createdAt: new Date("2025-10-10T08:30:00Z"),
  },
  {
    _id: "msg-002",
    name: "Lisa Martinez",
    email: "lisa.m@filmproduction.com",
    subject: "Integration with ftrack",
    message: "Does your platform integrate with ftrack? We're using it for project management and would like to know about compatibility.",
    createdAt: new Date("2025-10-12T14:15:00Z"),
  },
  {
    _id: "msg-003",
    name: "Robert Lee",
    email: "robert.lee@studio.com",
    subject: "Bug report - upload feature",
    message: "I'm experiencing issues when uploading large video files (>2GB). The upload seems to timeout. Can you look into this?",
    createdAt: new Date("2025-10-14T11:45:00Z"),
  },
];

export const mockFeedback = [
  {
    _id: "feedback-001",
    accountId: "507f1f77bcf86cd799439012",
    rating: 5,
    feedbackText: "The complexity analysis is incredibly accurate and has saved our team countless hours in estimation. Highly recommend!",
    featureSuggestions: ["Batch processing for multiple shots", "Export reports to PDF"],
    createdAt: new Date("2025-09-15T10:00:00Z"),
  },
  {
    _id: "feedback-002",
    accountId: "507f1f77bcf86cd799439014",
    rating: 4,
    feedbackText: "Great tool overall. The UI is clean and intuitive. Would love to see more customization options for the dashboard.",
    featureSuggestions: ["Custom dashboard widgets", "Dark mode", "Mobile app"],
    createdAt: new Date("2025-09-28T16:30:00Z"),
  },
  {
    _id: "feedback-003",
    accountId: "507f1f77bcf86cd799439013",
    rating: 5,
    feedbackText: "The VFX hour predictions are spot-on! This has revolutionized how we bid on projects.",
    featureSuggestions: ["API access for custom integrations", "Real-time collaboration features"],
    createdAt: new Date("2025-10-05T09:45:00Z"),
  },
];

// Mock API token for demonstration
export const mockToken = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOiI1MDdmMWY3N2JjZjg2Y2Q3OTk0MzkwMTEiLCJyb2xlIjoiYWRtaW4iLCJpYXQiOjE2OTcwMDAwMDAsImV4cCI6MTk5NzAwMDAwMH0.dGhpc19pc19hX21vY2tfdG9rZW5fZm9yX2RlbW9fcHVycG9zZXM";

// Mock credentials for demo login
export const mockCredentials = {
  email: "john.doe@vfxstudio.com",
  password: "Demo123!",
};

// Function to simulate API delay
export const simulateDelay = (ms: number = 500) => 
  new Promise(resolve => setTimeout(resolve, ms));

// Mock API responses
export const mockApiResponses = {
  login: async (email: string, password: string) => {
    await simulateDelay();
    if (email === mockCredentials.email && password === mockCredentials.password) {
      return {
        token: mockToken,
        data: mockAccounts[0],
      };
    }
    throw new Error("Invalid credentials");
  },
  
  getAccount: async (token: string) => {
    await simulateDelay();
    if (token === mockToken) {
      return { data: mockAccounts[0] };
    }
    throw new Error("Invalid token");
  },
  
  getResults: async (project?: string, sequence?: string) => {
    await simulateDelay();
    let results = [...mockResults];
    if (project) {
      results = results.filter(r => r.project === project);
    }
    if (sequence) {
      results = results.filter(r => r.sequence === sequence);
    }
    return { data: results };
  },
  
  getProjects: async (organizationId?: string) => {
    await simulateDelay();
    let projects = [...mockProjects];
    if (organizationId) {
      projects = projects.filter(p => p.organizationId === organizationId);
    }
    return { data: projects };
  },
  
  getCollaborators: async (organizationName: string) => {
    await simulateDelay();
    const collaborators = mockAccounts.filter(
      acc => acc.organizationName === organizationName
    );
    return { data: collaborators };
  },
  
  getContactMessages: async () => {
    await simulateDelay();
    return { data: mockContactMessages };
  },
  
  getFeedback: async () => {
    await simulateDelay();
    return { data: mockFeedback };
  },
};

export default {
  mockAccounts,
  mockProjects,
  mockSequences,
  mockResults,
  mockContactMessages,
  mockFeedback,
  mockToken,
  mockCredentials,
  simulateDelay,
  mockApiResponses,
};

