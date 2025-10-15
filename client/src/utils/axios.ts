import Axios from 'axios'
import { BACKEND_URL } from '../constants'
import MockApiService from './mockApiService'

const axios = Axios.create({
  baseURL: BACKEND_URL,
})

// Add response interceptor to handle API failures with mock data
axios.interceptors.response.use(
  (response) => response,
  async (error) => {
    const { config } = error;
    
    // If the request failed and we haven't already retried with mock data
    if (!config._mockRetry) {
      config._mockRetry = true;
      
      console.warn('API call failed, using mock data instead:', error.message);
      MockApiService.enableMockMode();
      
      try {
        // Return mock data based on the request
        const mockResponse = await MockApiService.handleRequest(
          config,
          async () => { throw error; }
        );
        
        return {
          data: mockResponse,
          status: 200,
          statusText: 'OK (Mock Data)',
          headers: {},
          config,
        };
      } catch (mockError) {
        console.error('Mock data also failed:', mockError);
        return Promise.reject(error);
      }
    }
    
    return Promise.reject(error);
  }
);

export default axios
