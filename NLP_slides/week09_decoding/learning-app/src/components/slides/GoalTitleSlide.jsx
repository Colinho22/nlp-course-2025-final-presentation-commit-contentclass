/**
 * Goal Title Slide - Type 1
 * Large centered title for goal introduction
 */

import { Box, Typography, Paper, Chip } from '@mui/material';
import { motion } from 'framer-motion';

const GoalTitleSlide = ({ goal, slideNumber }) => {
  return (
    <motion.div
      initial={{ scale: 0.9, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '60vh' }}>
        <Paper elevation={3} sx={{ p: 6, textAlign: 'center', maxWidth: 700, bgcolor: `${goal.color}10` }}>
          <Typography variant="h1" sx={{ fontSize: '4rem', mb: 2 }}>
            {goal.icon}
          </Typography>
          <Typography variant="h3" color="primary" gutterBottom fontWeight="bold">
            {goal.title}
          </Typography>
          <Typography variant="h6" color="text.secondary" sx={{ mb: 3 }}>
            {goal.description}
          </Typography>
          <Chip label={`${goal.slides} slides`} color="primary" sx={{ mr: 1 }} />
          <Chip label={`Goal ${goal.id.replace('goal', '')}`} variant="outlined" />
        </Paper>
      </Box>
    </motion.div>
  );
};

export default GoalTitleSlide;
