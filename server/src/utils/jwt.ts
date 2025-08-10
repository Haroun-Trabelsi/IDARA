import jsonwebtoken from 'jsonwebtoken';
import { JWT_SECRET } from '../constants/index';

class JWT {
  instance: typeof jsonwebtoken = jsonwebtoken;
  secret: string;

  constructor() {
    this.secret = JWT_SECRET;
    console.log('Debug: JWT_SECRET value:', this.secret);
    if (!this.secret) {
      console.error('Debug: JWT_SECRET is not defined, authentication will fail!');
    }
    console.log('Debug: Server time during JWT initialization:', new Date().toISOString());
  }

  signToken(payload: Record<string, any>, expiresIn: jsonwebtoken.SignOptions['expiresIn'] = '4h') {
    if (!this.secret) {
      throw new Error('JWT_SECRET is not defined');
    }
    const token = this.instance.sign(payload, this.secret, { expiresIn });
    console.log('Debug: Payload utilisé:', payload);
    console.log('Debug: Signed token:', token);
    console.log('Debug: Token expiration:', new Date(Date.now() + (typeof expiresIn === 'string' ? this.parseExpiresIn(expiresIn) : expiresIn * 1000)).toISOString());
    return token;
  }

  verifyToken(token: string) {
    if (!this.secret) {
      throw new Error('JWT_SECRET is not defined');
    }
    try {
      const auth = this.instance.verify(token, this.secret);
      console.log('Debug: Verified token payload:', auth);
      return auth;
    } catch (error) {
      console.error('Debug: Token verification error:', (error as Error).message);
      return null;
    }
  }

  // Helper pour parser expiresIn (si besoin)
  private parseExpiresIn(expiresIn: string): number {
    const match = expiresIn.match(/^(\d+)([hdms])$/);
    if (!match) return 0;
    const value = parseInt(match[1], 10);
    const unit = match[2];
    switch (unit) {
      case 'h': return value * 60 * 60 * 1000; // heures
      case 'd': return value * 24 * 60 * 60 * 1000; // jours
      case 'm': return value * 60 * 1000; // minutes
      case 's': return value * 1000; // secondes
      default: return 0;
    }
  }
}

export default new JWT();
