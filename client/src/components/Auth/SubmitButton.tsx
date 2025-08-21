import React from 'react';
import { Button, CircularProgress } from '@mui/material';
import { Email as EmailIcon } from '@mui/icons-material';

interface SubmitButtonProps {
  loading: boolean;
  disabled: boolean;
  accountId: string | null;
}

const SubmitButton: React.FC<SubmitButtonProps> = ({ loading, disabled, accountId }) => (
  <Button
    type="submit"
    disabled={disabled || loading}
    fullWidth
    variant="contained"
    sx={{
      mb: 4,
      backgroundColor: '#4299e1',
      color: '#000',
      py: 1.75,
      fontWeight: 600,
      '&:hover': { backgroundColor: '#3182ce' },
      '&:disabled': { backgroundColor: '#333', color: '#666' },
    }}
    startIcon={
      loading ? <CircularProgress size={18} sx={{ color: '#666' }} /> : <EmailIcon />
    }
  >
    {loading ? 'Processing...' : (accountId ? 'Update Account' : 'Create Account')}
  </Button>
);

export default SubmitButton;
