import joi from 'joi';

class Joi {
  instance: typeof joi = joi;

  constructor() {}

  async validate(schema: Record<string, any>, body: Record<string, any>, options: joi.AsyncValidationOptions = {}) {
    try {
      await this.instance.object(schema).validateAsync(body, { ...options, abortEarly: false });
    } catch (error: any) {
      console.log('❌ Joi validation error:', error.message);
      return {
        statusCode: 400,
        message: error.details.map((d: any) => d.message).join(', '),
      };
    }
  }

  get profileSchema() {
    return {
      name: this.instance.string().regex(/^[a-zA-Z0-9]+$/).min(2).required().messages({
        'string.pattern.base': 'Name must contain only letters and numbers',
        'string.min': 'Name must be at least 2 characters long',
        'any.required': 'Name is required',
      }),
      surname: this.instance.string().regex(/^[a-zA-Z0-9]+$/).min(2).required().messages({
        'string.pattern.base': 'Surname must contain only letters and numbers',
        'string.min': 'Surname must be at least 2 characters long',
        'any.required': 'Surname is required',
      }),
      email: this.instance.string().email().required().messages({
        'string.email': 'Email must be a valid email address',
        'any.required': 'Email is required',
      }),
      teamSize: this.instance.string().required().messages({
        'any.required': 'Team size is required',
      }),
      region: this.instance.string().required().messages({
        'any.required': 'Region is required',
      }),
    };
  }

  get passwordSchema() {
    return {
      currentPassword: this.instance.string().min(8).required().messages({
        'string.min': 'Current password must be at least 8 characters long',
        'any.required': 'Current password is required',
      }),
      newPassword: this.instance.string().min(8)
        .regex(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/)
        .required()
        .messages({
          'string.min': 'New password must be at least 8 characters long',
          'string.pattern.base': 'New password must contain at least one uppercase letter, one number, and one special character (@$!%*?&)',
          'any.required': 'New password is required',
        }),
      confirmPassword: this.instance.string().valid(joi.ref('newPassword')).required().messages({
        'any.only': 'Passwords do not match',
        'any.required': 'Confirm password is required',
      }),
    };
  }
}

export default new Joi();