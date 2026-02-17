#with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        
            opt = self.optimizers()
            sch = self.lr_schedulers()
            self.lam = self.lam.to(self.device)
            self.model = self.model.eval()
            x = self.x
            types = self.types
            
            mols, all_mols, log_probs, anchor_steps1, sim_first1, sim_anchor1, sim_last1 = pipeline_with_logprob(
                self.model,
                x, types,
                num_inference_steps=args.sample_steps,
                scheduler = self.scheduler,
                B = self.args.batch_size,
                device=self.device,
                eta = self.eta
            )
            mols2, all_mols2, log_probs2, anchor_steps2, sim_first2, sim_anchor2, sim_last2 = pipeline_with_logprob(
                self.model,
                x, types,
                num_inference_steps=args.sample_steps,
                scheduler = self.scheduler,
                B = self.args.batch_size,
                device=self.device,
                eta = self.eta
            )
            coords1, atoms1 = zip(*all_mols)
            coords2, atoms2 = zip(*all_mols2)
            coords1 = torch.stack(list(coords1), dim=-1).permute(0, 3, 1, 2)
            atoms1 = torch.stack(list(atoms1), dim=-1).permute(0, 3, 1, 2)
            coords2 = torch.stack(list(coords2), dim=-1).permute(0, 3, 1, 2)
            atoms2 = torch.stack(list(atoms2), dim=-1).permute(0, 3, 1, 2)
            log_probs_coord1, log_probs_atom1 = torch.split(log_probs, 29, -1)
            log_probs_coord2, log_probs_atom2 = torch.split(log_probs2, 29, -1)
            del all_mols, all_mols2, log_probs, log_probs2
            # compute rewards asynchronously
            rewards1 = get_reward(mols, self.rewarder, self.vocab, self.args.vdW)
            rewards2 = get_reward(mols2, self.rewarder, self.vocab, self.args.vdW)
            #logging rewards
            rewards1 = rewards1.transpose(0, 1).to(self.device)
            rewardsT_1, rewards_anchor_1, rewards0_1 = rewards1.split(1, 1)
            rewards0_1 = torch.mean(rewards0_1)
            rewards_anchor_1 = torch.mean(rewards_anchor_1)
            rewardsT_1 = torch.mean(rewardsT_1)

            rewards2 = rewards2.transpose(0, 1).to(self.device)
            rewardsT_2, rewards_anchor_2, rewards0_2 = rewards2.split(1, 1)
            rewards0_2 = torch.mean(rewards0_2)
            rewards_anchor_2 = torch.mean(rewards_anchor_2)
            rewardsT_2 = torch.mean(rewardsT_2)

            reward0_mean = (rewards0_1 + rewards0_2)/2
            self.log('Rewards 1 step 0', rewards0_1)
            self.log('Rewards 1 anchor', rewards_anchor_1)
            self.log('Rewards 1 last step', rewardsT_1)
            self.log('Rewards 2 step 0', rewards0_2)
            self.log('Rewards 2 anchor', rewards_anchor_2)
            self.log('Rewards 2 last step', rewardsT_2)
            self.log('Reward0_mean', reward0_mean.item(), on_step=True)
            #del rewardsT_1, rewards_anchor_1, rewards0_1, rewardsT_2, rewards_anchor_2, rewards0_2
            
            sample ={
                    'coords1': coords1[:, :-1],
                    'atoms1': atoms1[:, :-1],
                    'coords2': coords2[:, :-1],
                    'atoms2': atoms2[:, :-1],
                    'next_coords1': coords1[:, 1:],
                    'next_atoms1': atoms1[:, 1:],
                    'next_coords2': coords2[:, 1:],
                    'next_atoms2': atoms2[:, 1:],
                    "log_probs_coord1": log_probs_coord1,
                    "log_probs_atom1": log_probs_atom1,
                    "log_probs_coord2": log_probs_coord2,
                    "log_probs_atom2": log_probs_atom2,
                    "rewards1": rewards1,
                    "rewards2": rewards2,
                    "anchor_steps1": anchor_steps1,
                    "sim_first1": sim_first1,
                    "sim_anchor1": sim_anchor1,
                    "sim_last1": sim_last1,
                    "anchor_steps2": anchor_steps2,
                    "sim_first2": sim_first2,
                    "sim_anchor2": sim_anchor2,
                    "sim_last2": sim_last2,
                }
            sample["rewards1"] = (
                rewards1[:, 0:1].repeat(1, args.sample_steps) * sample["sim_last1"]
                + rewards1[:, 1:-1].repeat(1, args.sample_steps) * sample["sim_anchor1"]
                + rewards1[:, -1:].repeat(1, args.sample_steps) * sample["sim_first1"]
            ) / (sample["sim_first1"] + sample["sim_anchor1"] + sample["sim_last1"])
            sample["rewards1"][:, 0], sample["rewards1"][:, -1] = rewards1[:, 0], rewards1[:, -1]
            sample["rewards1"][torch.arange(args.batch_size, device=self.device), sample["anchor_steps1"]] = rewards1[:, 1]

            sample["rewards2"] = (
                rewards2[:, 0:1].repeat(1, args.sample_steps) * sample["sim_last2"]
                + rewards2[:, 1:-1].repeat(1, args.sample_steps) * sample["sim_anchor2"]
                + rewards2[:, -1:].repeat(1, args.sample_steps) * sample["sim_first2"]
            ) / (sample["sim_first2"] + sample["sim_anchor2"] + sample["sim_last2"])
            sample["rewards2"][:, 0], sample["rewards2"][:, -1] = rewards2[:, 0], rewards2[:, -1]
            sample["rewards2"][torch.arange(args.batch_size, device=self.device), sample["anchor_steps2"]] = rewards2[:, 1]

            sample["rewards1"] = sample["rewards1"].detach()
            sample["rewards2"] = sample["rewards2"].detach()

            for t in reversed(range(self.args.sample_steps)):
                next_return = sample['rewards1'][:, t + 1] if t < self.args.sample_steps - 1 else 0.0
                sample['rewards1'][:, t] += 0.99 * next_return
                next_return = sample['rewards2'][:, t + 1] if t < self.args.sample_steps - 1 else 0.0
                sample['rewards2'][:, t] += 0.99 * next_return


            del sample["anchor_steps1"], sample["sim_first1"], sample["sim_anchor1"], sample["sim_last1"]
            del sample["anchor_steps2"], sample["sim_first2"], sample["sim_anchor2"], sample["sim_last2"]
            # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
            #################### TRAINING ####################

            self.model = self.model.train()
    
            timesteps = self.scheduler.timesteps.repeat(1, 1)
            for inner_epoch in range(args.inner_epochs):
                # rebatch for training            # train
                # Assuming the rewards are calculated and available in the original sparse format:
    # rewards1: (Batch Size, 3) where columns are [R_last, R_anchor, R_start]
    # rewards2: (Batch Size, 3)

            
    # --- REWARD SIGNAL DIAGNOSTICS END ---
                adv = torch.cat((sample["rewards1"], sample["rewards2"]))
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                adv1, adv2 = torch.chunk(adv, chunks=2)
                #adv1 = torch.clamp(adv1, -self.adv_clip_max, self.adv_clip_max)
                #adv2 = torch.clamp(adv2, -self.adv_clip_max, self.adv_clip_max)
                advantage_diff = self.advantage_scale * (adv1 - adv2)
                
                # --- REWARD SIGNAL DIAGNOSTICS START (Step T) ---
                
                ld = []
                ad = []
                l = 0.0
                for j in range(self.args.sample_steps):
                    atoms1 = sample["atoms1"][:, j]
                    atoms2 = sample["atoms2"][:, j]
                    coords1 = sample["coords1"][:, j]
                    coords2 = sample["coords2"][:, j]
                    next_atoms1 = sample["next_atoms1"][:, j]
                    next_atoms2 = sample["next_atoms2"][:, j]
                    next_coords1 = sample["next_coords1"][:, j]
                    next_coords2 = sample["next_coords2"][:, j]
                    logp_coord1_old = sample["log_probs_coord1"][:, j].detach()
                    logp_atom1_old  = sample["log_probs_atom1"][:, j].detach()
                    logp_coord2_old = sample["log_probs_coord2"][:, j].detach()
                    logp_atom2_old  = sample["log_probs_atom2"][:, j].detach()
                    adv_diff = adv1[:, j] - adv2[:, j]
                    coord1, types1= self.model( 
                        atoms1,
                        coords1,
                        timesteps[:, j]
                    )
                    coord2, types2= self.model( 
                        atoms2,
                        coords2,
                        timesteps[:, j]
                    )
                    # compute the log-prob of next_latents given latents under the current modelù
                    
                    _, _, log_prob_coord1 = ddim_step_with_logprob(
                        self.scheduler,
                        coord1,      # predicted ε OR x0 OR v
                        timesteps[:, j],                 # current timestep
                        coords1,               # noisy sample at time t
                        eta=self.eta,
                        x_prev=next_coords1
                    )
                    t1, _, log_prob_types1 = categorical_reverse_step(
                        self.scheduler,
                        types1,
                        timesteps[:, j],   # predicted ε OR x0 OR v            
                        atoms1,
                        eta=self.eta,
                        x_prev=next_atoms1,
                    )

                    _, _, log_prob_coord2 = ddim_step_with_logprob(
                        self.scheduler,
                        coord2,      # predicted ε OR x0 OR v
                        timesteps[:, j],                 # current timestep
                        coords2,
                        eta=self.eta,
                        x_prev=next_coords2
                    )
                    t2, _, log_prob_types2 = categorical_reverse_step(
                        self.scheduler,
                        types2,
                        timesteps[:, j],   # predicted ε OR x0 OR v              
                        atoms2,
                        eta=self.eta,
                        x_prev=next_atoms2
                    )
                    logc1 = torch.clamp(log_prob_coord1, logp_coord1_old - self.clip_range, logp_coord1_old + self.clip_range)
                    logc2 = torch.clamp(log_prob_coord2, logp_coord2_old - self.clip_range, logp_coord2_old + self.clip_range)
                    loga1 = torch.clamp(log_prob_types1, logp_atom1_old - self.clip_range, logp_atom1_old + self.clip_range)
                    loga2 = torch.clamp(log_prob_types2, logp_atom2_old - self.clip_range, logp_atom2_old + self.clip_range)
                    
                    log_ratio1 = (log_prob_coord1 - logp_coord1_old) +( log_prob_types1 - logp_atom1_old)
                    log_ratio2 = (log_prob_coord2 - logp_coord2_old) + (log_prob_types2 - logp_atom2_old)
                    log_diff = log_ratio1 - log_ratio2
                    log_ratio1_clipped = (logc1 - logp_coord1_old) +( loga1 - logp_atom1_old)
                    log_ratio2_clipped = (logc2 - logp_coord1_old) +( loga2 - logp_atom1_old)
                    log_diff_clipped = log_ratio1_clipped - log_ratio2_clipped
                   
                    adv_diff=adv_diff.unsqueeze(-1)
                    log_weights = (self.lam[:, j] / self.args.log_scale).unsqueeze(-1)
                    log_diff_clipped = log_weights * log_diff_clipped 
                    loss_clipped =  torch.square(log_diff_clipped - adv_diff)
                    log_diff = log_weights * log_diff
                    loss = torch.square(log_diff - adv_diff)
                    loss_sdpo = torch.mean(torch.maximum(loss_clipped, loss)) / self.accumulation_steps# The SDPO term
                    l += loss_sdpo
                    if self.args.debug: # and (self.global_step % 100) == 0: # Only check every 100 steps
                        # --- 1. RAW ADVANTAGE DIFFERENCE CHECK (Before Normalization) ---

                        print(f"\n--- Signal Diagnostics (Step {self.global_step}) ---")
                        print("\n[Raw Advantage Diff (A_diff) Statistics - Before Normalization]")
                        print(f"Adv Diff Mean (Raw): {adv_diff.mean().item():.6f}")
                        print(f"Adv Diff Std (Raw):  {adv_diff.std().item():.6f}")
                        print(f"Adv Diff Min/Max:    [{adv_diff.min().item():.4f}, {adv_diff.max().item():.4f}]")
                        print('Rewards 1 step 0', rewards0_1)
                        print('Rewards 1 anchor', rewards_anchor_1)
                        print('Rewards 1 last step', rewardsT_1)
                        print('Rewards 2 step 0', rewards0_2)
                        print('Rewards 2 anchor', rewards_anchor_2)
                        print('Rewards 2 last step', rewardsT_2)
                        # Interpretation: If Std is near zero (e.g., < 1e-4), the reward signal is dead.
                        
                        # --- 2. REWARD VARIANCE CHECK ---
                        # We check the variance of the sparse rewards across the batch
                        
                        # Use rewards1 for component analysis
                        R_start_1 = rewards1[:, -1]-1
                        R_anchor_1 = rewards1[:, 1]
                        R_last_1 = rewards1[:, 0]
                        l_diff = torch.maximum(log_diff, log_diff_clipped)
                        print("\n[Sparse Reward R1 StdDev Check (Across Batch)]")
                        print(f"R_Start StdDev:   {R_start_1.std().item():.6f}")
                        print(f"R_Anchor StdDev:  {R_anchor_1.std().item():.6f}")
                        print(f"R_Last StdDev:    {R_last_1.std().item():.6f}")
                        print(f"Log_Diff Mean: {l_diff.mean().item():.4f}")
                        print(f"Log_Diff Std:  {l_diff.std().item():.4f}")
                        print(f"Log_Diff Min:  {l_diff.min().item():.4f}")
                        print(f"Log_Diff Max:  {l_diff.max().item():.4f}")
                        print(f"Log_Diff Var:  {l_diff.var().item():.4f}")
                        print(f"Adv Var:  {adv_diff.var().item():.4f}")
                        print(f"Log_Prob Grad Fn: {log_prob_coord1.grad_fn}")
                        print(f"Requires Grad: {coord1.requires_grad}")
                        print("---------------------------------------------------")
                        
                    ld.append(log_diff.mean())
                    ad.append(adv_diff.mean())
                
                self.manual_backward(l) 
                if self.args.debug:
                        total_norm = 0.0
                        for n, p in self.model.named_parameters():
                            if p.grad is not None:
                                param_norm = p.grad.detach().data.norm(2)
                                total_norm += param_norm.item() ** 2
                            else:
                                print(n)
                            
                        total_norm = total_norm ** 0.5

                        # Log to your tracker
                        print(total_norm)
                if self.accumulation == self.accumulation_steps:
                    self.accumulation = 0
                    self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
                    opt.step()
                    sch.step()
                    opt.zero_grad()
                else:
                    self.accumulation += 1
                l+=loss_sdpo.detach().item()
                ld = torch.stack(ld).flatten()
                ad = torch.stack(ad).flatten()

                cos_sim = torch.corrcoef(torch.stack([ld, ad]))[0, 1]
                ts = self.args.sample_steps
                corr_mid = torch.corrcoef(torch.stack([ld[ts//4:ts//4*3], ad[ts//4:3*ts//4]]))[0, 1]
                corr_early = torch.corrcoef(torch.stack([ld[:ts//4], ad[:ts//4]]))[0, 1]
                corr_late = torch.corrcoef(torch.stack([ld[3*ts//4:], ad[3*ts//4:]]))[0, 1]
                # cos_sim is now a vector of size [T] representing correlation at each ste
                self.log("corr", cos_sim)
                self.log_dict({'corr_early': corr_early, 'corr_mid': corr_mid, 'corr_late':corr_late})
                #loss_unclipped = torch.square(log_weights * log_diff + advantages).mean(0)
                #loss_clipped = torch.square(log_weights * log_diff_clipped + advantages).mean(0)
                self.log('log_diff', ld.mean())
                self.log('advantage_diff', ad.mean())

                # --- 4. Adaptive Beta Adjustment (After loss calculation, before optimizer step) --
                # ... (Your logging and return statement using total_loss) ...
                self.log('Training_loss', l)
                if self.best_loss:
                    if l < self.best_loss:
                        self.best_loss = l
                        path = f'logs/TrainingSDPO/ckpts/Finetuned-step-{self.global_step}-loss-{l}'
                        torch.save(self.model, path)
                        self.paths.append(path)
                        if len(self.paths) > 5:
                            path = self.paths.pop(0)
                            os.remove(path)
                else:
                    self.best_loss = l
            return torch.tensor(l)
        #print(prof.key_averages().table())