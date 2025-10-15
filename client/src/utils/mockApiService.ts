/**
 * Mock API Service
 * This service provides fallback functionality when the real API is unavailable.
 * It intercepts failed API calls and returns mock data instead.
 */

import { mockApiResponses } from './mockData';

export const useMockData = true; // Set to false to always try real API first

export class MockApiService {
  private static instance: MockApiService;
  private mockMode: boolean = false;

  private constructor() {}

  static getInstance(): MockApiService {
    if (!MockApiService.instance) {
      MockApiService.instance = new MockApiService();
    }
    return MockApiService.instance;
  }

  enableMockMode() {
    this.mockMode = true;
    console.warn('🔄 Mock API Mode Enabled - Using dummy data for demonstration');
  }

  disableMockMode() {
    this.mockMode = false;
    console.info('✅ Mock API Mode Disabled - Using real API');
  }

  isMockMode(): boolean {
    return this.mockMode;
  }

  // Intercept and handle API calls with mock data
  async handleRequest(config: any, realApiCall: () => Promise<any>): Promise<any> {
    // If mock mode is disabled, try real API first
    if (!this.mockMode && !useMockData) {
      try {
        return await realApiCall();
      } catch (error) {
        console.warn('Real API call failed, falling back to mock data:', error);
        this.mockMode = true;
      }
    }

    // Use mock data
    return this.getMockResponse(config);
  }

  private async getMockResponse(config: any): Promise<any> {
    const { method, url, data } = config;
    
    try {
      // Auth endpoints
      if (url.includes('/auth/login') && method === 'post') {
        return await mockApiResponses.login(data.email, data.password);
      }
      
      if (url.includes('/auth/login') && method === 'get') {
        const token = config.headers?.Authorization?.replace('Bearer ', '');
        return await mockApiResponses.getAccount(token);
      }

      // Results endpoints
      if (url.includes('/results')) {
        const urlParams = new URLSearchParams(url.split('?')[1]);
        return await mockApiResponses.getResults(
          urlParams.get('project') || undefined,
          urlParams.get('sequence') || undefined
        );
      }

      // Projects endpoints
      if (url.includes('/projects')) {
        const urlParams = new URLSearchParams(url.split('?')[1]);
        return await mockApiResponses.getProjects(
          urlParams.get('organizationId') || undefined
        );
      }

      // Collaborators endpoints
      if (url.includes('/collaborators')) {
        return await mockApiResponses.getCollaborators('VFX Studio Alpha');
      }

      // Contact messages endpoints
      if (url.includes('/contact')) {
        return await mockApiResponses.getContactMessages();
      }

      // Feedback endpoints
      if (url.includes('/feedback')) {
        return await mockApiResponses.getFeedback();
      }

      // Default response for unmatched endpoints
      console.warn(`No mock data available for: ${method.toUpperCase()} ${url}`);
      return { data: [], message: 'Mock data not available for this endpoint' };
      
    } catch (error) {
      console.error('Error in mock API service:', error);
      throw error;
    }
  }
}

export default MockApiService.getInstance();

