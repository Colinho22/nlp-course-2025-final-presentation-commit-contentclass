/**
 * Framework Slide - Type 6
 * Shows conceptual framework with connected steps
 */

import { Box, Typography, Paper, Grid } from '@mui/material';
import { ArrowForward } from '@mui/icons-material';
import { motion } from 'framer-motion';

const FrameworkSlide = ({ title, steps, description }) => {
  return (
    <motion.div
      initial={{ y: 30, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <Box>
        <Typography variant="h4" color="primary" gutterBottom fontWeight="bold" sx={{ mb: 1 }}>
          {title}
        </Typography>
        {description && (
          <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
            {description}
          </Typography>
        )}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          {steps.map((step, i) => (
            <Box key={i} sx={{ display: 'flex', alignItems: 'center', flex: 1 }}>
              <Paper
                elevation={3}
                sx={{
                  p: 3,
                  flex: 1,
                  bgcolor: i === 0 ? 'secondary.light' : i === steps.length - 1 ? 'primary.main' : 'background.paper',
                  color: i === steps.length - 1 ? 'white' : 'text.primary',
                  textAlign: 'center'
                }}
              >
                <Typography variant="h6" fontWeight="bold" gutterBottom>
                  {step.title}
                </Typography>
                <Typography variant="body2">
                  {step.description}
                </Typography>
              </Paper>
              {i < steps.length - 1 && (
                <ArrowForward sx={{ mx: 1, fontSize: 40, color: 'primary.main' }} />
              )}
            </Box>
          ))}
        </Box>
      </Box>
    </motion.div>
  );
};

export default FrameworkSlide;
