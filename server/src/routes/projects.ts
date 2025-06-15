import express from 'express';
import { getFtrackSession } from '../utils/session';

const router = express.Router();
const session = getFtrackSession();

router.get('/projects', async (req, res) => {
    try {
        console.log('Connected to ftrack');
        console.log('About to query projects');
        const response = await (await session).query('select name from Project');
        const projects = response.data;
        
        console.info("Listing " + projects.length + " projects");
        console.log(projects.map((project) => project.name));
        
        // Send the project names as JSON array
        res.json(projects.map(project => project.name));
        
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Failed to fetch projects' });
    }
});

export default router;
