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

        res.json(projects.map(project => project.name));
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Failed to fetch projects' });
    }
});

// Get all tasks of project named "7760"
/*router.get('/tasks/:projectName', async (req, res) => {
    try {
        const projectName = req.params.projectName;

        console.log(`Fetching tasks for project: ${projectName}`);

        const sessionInstance = await session;

        // First, get the project entity
        const projectQuery = await sessionInstance.query(
            `select name, tasks from Project where name is "${projectName}"`
        );

        const project = projectQuery.data[0];
        if (!project) {
            return res.status(404).json({ error: 'Project not found' });
        }

        // Then, fetch all tasks linked to the project
        const taskQuery = await sessionInstance.query(
            `select name, status, assignee from Task where project.name is "${projectName}"`
        );

        const tasks = taskQuery.data;

        console.info(`Found ${tasks.length} tasks`);
        res.json(tasks);

    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Failed to fetch tasks' });
    }
});*/

export default router;
