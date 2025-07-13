import joi from 'joi'

class Joi {
  instance: typeof joi = joi

  constructor() {}

  async validate(schema: Record<string, any>, body: Record<string, any>, options: joi.AsyncValidationOptions = {}) {
    try {
      await this.instance.object(schema).validateAsync(body, { ...options, abortEarly: false })
    } catch (error: any) {
      console.log('❌ Joi validation error:', error.message)
      return {
        statusCode: 400,
        message: error.message,
      }
    }
  }
}

export default new Joi()