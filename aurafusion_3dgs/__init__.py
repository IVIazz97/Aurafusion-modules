# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AuraFusion360 (Wu et al., CVPR 2025) object removal + inpainting on a plain graphdeco 3DGS.

Input: point_cloud.ply + its COLMAP scene + 2D object masks. Output: an inpainted point_cloud.ply.
"""
