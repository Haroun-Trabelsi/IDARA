import express from 'express';
const { sendMessage, getMessages } = require('../controllers/contact/contactController');

const router = express.Router();

router.post('/', sendMessage);
router.get('/', getMessages);

export default router;