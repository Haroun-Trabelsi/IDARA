export const validateProfile = (data: { name: string; surname: string; email: string; teamSize?: string; region?: string; organizationName?: string }) => {
  let isValid = true;
  const errors = { name: '', surname: '', email: '', teamSize: '', region: '', organizationName: '' };

  if (!data.name || !/^[a-zA-Z0-9\s]+$/.test(data.name) || data.name.length < 2) {
    errors.name = 'Name must contain only letters, numbers, and spaces and be at least 2 characters long';
    isValid = false;
  }
  if (!data.surname || !/^[a-zA-Z0-9\s]+$/.test(data.surname) || data.surname.length < 2) {
    errors.surname = 'Surname must contain only letters, numbers, and spaces and be at least 2 characters long';
    isValid = false;
  }
  if (!data.email || !/^\S+@\S+\.\S+$/.test(data.email)) {
    errors.email = 'Email must be a valid email address';
    isValid = false;
  }
  if (data.teamSize !== undefined && !data.teamSize) {
    errors.teamSize = 'Team size is required';
    isValid = false;
  }
  if (data.region !== undefined && !data.region) {
    errors.region = 'Region is required';
    isValid = false;
  }
  if (data.organizationName !== undefined && (!data.organizationName || !/^[a-zA-Z0-9\s]+$/.test(data.organizationName) || data.organizationName.length < 2)) {
    errors.organizationName = 'Organization name must contain only letters, numbers, and spaces and be at least 2 characters long';
    isValid = false;
  }

  return { isValid, errors };
};

export const validatePassword = (data: { currentPassword: string; newPassword: string; confirmPassword: string }) => {
  let isValid = true;
  const errors = { currentPassword: '', newPassword: '', confirmPassword: '' };

  if (!data.currentPassword || data.currentPassword.length < 8) {
    errors.currentPassword = 'Current password must be at least 8 characters long';
    isValid = false;
  }
  if (!data.newPassword || !/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/.test(data.newPassword)) {
    errors.newPassword = 'New password must contain at least one uppercase letter, one number, and one special character (@$!%*?&)';
    isValid = false;
  }
  if (data.newPassword !== data.confirmPassword) {
    errors.confirmPassword = 'Passwords do not match';
    isValid = false;
  }

  return { isValid, errors };
};