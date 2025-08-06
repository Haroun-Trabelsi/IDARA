import { type RequestHandler } from 'express';
import jwt from '../utils/jwt';

const checkBearerToken: RequestHandler = (req, res, next) => {
  try {
    const token = req.headers.authorization?.split(' ')[1];
    console.log('Debug: Authorization header received:', req.headers.authorization);
    console.log('Debug: Extracted token:', token);

    if (!token) {
      console.log('Debug: Token not provided');
      return next({
        statusCode: 400,
        message: 'Token not provided',
      });
    }

    const auth = jwt.verifyToken(token);
    console.log('Debug: Verification result:', auth);

    if (!auth) {
      console.log('Debug: Invalid token');
      return next({
        statusCode: 401,
        message: 'Invalid token',
      });
    }

    req.auth = typeof auth === 'string' ? JSON.parse(auth) : auth;
    console.log('Debug: req.auth set to:', req.auth);

    next();
  } catch (error) {
    console.log('Debug: Error in checkBearerToken:', (error instanceof Error ? error.message : 'Unknown error'));
    next({
      statusCode: 401,
      message: 'Invalid token',
    });
  }
};

export default checkBearerToken;