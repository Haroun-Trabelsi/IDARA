import express from 'express';
const {  getMessages } = require('../controllers/contact/contactController');

const router = express.Router();

// Route protégée pour admin seulement
router.get('/contact-messages' , getMessages);

export default router;