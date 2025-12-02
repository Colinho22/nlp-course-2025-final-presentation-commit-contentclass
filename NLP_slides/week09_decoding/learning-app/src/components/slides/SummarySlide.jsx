/**
 * Summary Slide - Type 4
 * Key points with checkmark icons
 */

import { Box, Typography, List, ListItem, ListItemIcon, ListItemText, Paper } from '@mui/material';
import { CheckCircle } from '@mui/icons-material';
import { motion } from 'framer-motion';

const SummarySlide = ({ title, points, highlight }) => {
  return (
    <motion.div
      initial={{ y: 20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.4 }}
    >
      <Box>
        <Typography variant="h4" color="primary" gutterBottom fontWeight="bold" sx={{ mb: 3 }}>
          {title}
        </Typography>
        <Paper elevation={1} sx={{ p: 3 }}>
          <List>
            {points.map((point, i) => (
              <ListItem key={i}>
                <ListItemIcon>
                  <CheckCircle color="success" />
                </ListItemIcon>
                <ListItemText
                  primary={point}
                  primaryTypographyProps={{ variant: 'body1', fontWeight: 500 }}
                />
              </ListItem>
            ))}
          </List>
          {highlight && (
            <Box sx={{ mt: 3, p: 2, bgcolor: 'warning.light', borderRadius: 1 }}>
              <Typography variant="body1" fontWeight="bold">
                💡 Key Insight: {highlight}
              </Typography>
            </Box>
          )}
        </Paper>
      </Box>
    </motion.div>
  );
};

export default SummarySlide;
