// gmmbin/gmm-align.cc

// Copyright 2009-2012  Microsoft Corporation
//           2012-2014 Johns Hopkins University (Author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model-2D.h"
#include "gmm/decodable-am-diag-gmm.h"

namespace kaldi {
	// this function returns the logprob that 'data' is emitted from 'pdf'
	// Refered to gmm/decodable-am-diag-gmm.cc:28 ( LogLikelihoodZeroBased ) 
	BaseFloat LogLikelihood_Temp(const DiagGmm &pdf, const VectorBase<BaseFloat> &data) {
		// check if everything is in order
		if (pdf.Dim() != data.Dim()) {
			KALDI_ERR << "Dim mismatch: data dim = " << data.Dim()
				<< " vs. model dim = " << pdf.Dim();
		}
		// Should check before entering this function ! ! 
		if (!pdf.valid_gconsts()) {
			KALDI_ERR << ": Must call ComputeGconsts() before computing likelihood.";
		}

		BaseFloat log_sum_exp_prune = -1.0;

		Vector<BaseFloat> data_squared;
		data_squared.CopyFromVec(data);
		data_squared.ApplyPow(2.0);

		Vector<BaseFloat> loglikes(pdf.gconsts());  // need to recreate for each pdf
		// loglikes +=  means * inv(vars) * data.
		loglikes.AddMatVec(1.0, pdf.means_invvars(), kNoTrans, data, 1.0);
		// loglikes += -0.5 * inv(vars) * data_sq.
		loglikes.AddMatVec(-0.5, pdf.inv_vars(), kNoTrans, data_squared, 1.0);

		BaseFloat log_sum = loglikes.LogSumExp(log_sum_exp_prune);
		if (KALDI_ISNAN(log_sum) || KALDI_ISINF(log_sum))
			KALDI_ERR << "Invalid answer (overflow or invalid variances/features?)";

		return log_sum;
	}

} // end namespace kaldi


int main(int argc, char *argv[]) {
	try {
		using namespace kaldi;
		typedef kaldi::int32 int32;

		const char *usage =
			"Align features given [GMM-based] models.\n"
			"Usage:   gmm-align-2D [options] <model-in> <feature-rspecifier> "
			"<block-rspecifier> <transcriptions-rspecifier> <alignments-wspecifier>\n"
			"e.g.: \n"
			" gmm-align 1.mdl block  scp:train.scp "
			"'ark:sym2int.pl -f 2- words.txt text|' ark:1.ali\n";
		ParseOptions po(usage);
		BaseFloat acoustic_scale = 1.0;

		po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");

		po.Read(argc, argv);

		if (po.NumArgs() != 5) {
			po.PrintUsage();
			exit(1);
		}

		std::string model_in_filename = po.GetArg(1);
		std::string feature_rspecifier = po.GetArg(2);
		std::string block_rspecifier = po.GetArg(3);
		std::string transcript_rspecifier = po.GetArg(4);
		std::string alignment_wspecifier = po.GetArg(5);

		TransitionModel_2D trans_model;
		AmDiagGmm am_gmm;
		{
			bool binary;
			Input ki(model_in_filename, &binary);
			trans_model.Read(ki.Stream(), binary);
			am_gmm.Read(ki.Stream(), binary);
		}
		const HmmTopology_2D topo = trans_model.GetTopo();

		SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
		RandomAccessInt32VectorReader block_info_reader(block_rspecifier);
		RandomAccessInt32VectorReader transcript_reader(transcript_rspecifier);
		Int32VectorWriter alignment_writer(alignment_wspecifier);

		int32 num_done = 0, num_err = 0;
		double tot_like = 0.0;
		kaldi::int64 frame_count = 0;
		for (; !feature_reader.Done(); feature_reader.Next()) {
			std::string utt = feature_reader.Key();
			const Matrix<BaseFloat> &features = feature_reader.Value();
			if (features.NumRows() == 0) {
				KALDI_WARN << "Zero-length features for utterance: " << utt;
				num_err++;
				continue;
			}
			// ������sampleû��text��Ϣ������
			if (!transcript_reader.HasKey(utt)) {
				KALDI_WARN << "No transcript found for utterance " << utt;
				num_err++;
				continue;
			}
			// ������sampleû��block��Ϣ������
			if (!block_info_reader.HasKey(utt)) {
				KALDI_WARN << "No block info found for utterance " << utt;
				num_err++;
				continue;
			}
			const std::vector<int32> &block_info = block_info_reader.Value(utt);
			// ���ĳ��sample��block-information����������row_num col_num tol_num�����������+1���������sample
			if (block_info.size() != 3) {
				KALDI_WARN << "Block information size supposed to be 3, but get "
					<< block_info.size() << " from " << utt << " instead.";
				num_err++;
				continue;
			}
			size_t block_row_num = static_cast<size_t>(block_info[0]);
			size_t block_col_num = static_cast<size_t>(block_info[1]);
			size_t block_tol_num = static_cast<size_t>(block_info[2]);
			// ���ĳ��sample��block-information����ȷ��row_num*col_num��=tol_num�����������+1���������sample
			if (block_tol_num != block_row_num * block_col_num) {
				KALDI_WARN << "Block rows number * cols number not equal to total number, sample name:" << utt
					<< "rows:" << block_row_num << " cols:" << block_col_num << " total:" << block_tol_num;
				num_err++;
				continue;
			}
			const std::vector<int32> &transcript = transcript_reader.Value(utt);
			//TODO:��ʱ�����ܳ���һ��phone��sample
			if (transcript.size() != 1) {
				KALDI_WARN << "Currently could only handle single phone for each sample, but get "
					<< transcript.size() << " phones for " << utt;
				num_err++;
				continue;
			}
			int32 phone_id = transcript[0]; //��������Ӧ��phone
			const HmmTopology_2D::TopologyEntry_2D &entry = topo.TopologyForPhone(phone_id);//entry for this phone
			size_t state_row_num = static_cast<size_t>(topo.TopologyShapeRowsForPhone(phone_id)),//��phone��Ӧ��state-map������
				state_col_num = static_cast<size_t>(topo.TopologyShapeColsForPhone(phone_id)),//��phone��Ӧ��state-map������
				state_tol_num = entry.size() - 1, // how many emitting HmmStates that this entry has.
				hmm_state; //ĳһblock��Ӧ��state�ڸ�phone�е�index��0-based
			int32 trans_state; //��state������state�е����

			KALDI_ASSERT(state_tol_num == state_row_num * state_col_num);
			/* explanation of three most important matrixs in-order
				// Store the biggest prob of max_{StateOf_LeftBlock, StateOf_TopBlock}{P[block_row_emitted_by_state_col, O_past | Gmm-Hmm model]}
				// Store argmax_{state of left block}{when block row is emitted by state col}   [first col should be none-sense]
				// Store argmax_{state of top block}{when block row is emitted by state col}   [first row should be none-sense]
			*/
			std::vector< std::vector<BaseFloat> > log_delta_matrix(block_tol_num, std::vector<BaseFloat>(state_tol_num, 0.0));
			std::vector< std::vector<size_t> > most_like_state_top(block_tol_num, std::vector<size_t>(state_tol_num, 0));//��ĳһblock��ĳһstate����ʱ����һblock�Ϸ���block��Ӧ��state�������ʲô
			std::vector< std::vector<size_t> > most_like_state_left(block_tol_num, std::vector<size_t>(state_tol_num, 0));//��ĳһblock��ĳһstate����ʱ����һblock����block��Ӧ��state�������ʲô
			
			// ���ó�ʼ����
			std::vector<BaseFloat> log_pai_top_down(state_tol_num, Log(0.1)); // the prob that the first block row emitted by every states in this phone
			std::vector<BaseFloat> log_pai_left_right(state_tol_num, Log(0.1)); // the prob that the first block col emitted by every states in this phone
			BaseFloat first_row_state_pai = Log(1.0 / static_cast<BaseFloat>(state_col_num)),
				      first_col_state_pai = Log(1.0 / static_cast<BaseFloat>(state_row_num));
			for (size_t i = 0; i < state_col_num; i++) 
				log_pai_top_down[i] = first_row_state_pai;
			for (size_t i = 0; i < state_tol_num; i += state_col_num)
				log_pai_left_right[i] = first_col_state_pai;

			std::vector< std::vector<BaseFloat> > log_trans_prob_top_down(state_tol_num, std::vector<BaseFloat>(state_tol_num, 0.0));
			std::vector< std::vector<BaseFloat> > log_trans_prob_left_right(state_tol_num, std::vector<BaseFloat>(state_tol_num, 0.0));
			BaseFloat log_prob_no_trans = -10000; //�����ڶ�Ӧ��ת�ƻ�ʱ�Ķ���ת�Ƹ��ʣ����㵽����ת�Ƹ���Ϊ exp{ -10000 } ~= 0
			for (size_t from = 0; from < state_tol_num; from++) {
				for (size_t to = 0; to < state_tol_num; to++) {
					int32 trans_state_from = trans_model.PairToState(phone_id, from);
					int32 trans_state_to = trans_model.PairToState(phone_id, to);
					int32 trans_id = trans_model.StatePairToTransitionId_TopDown(trans_state_from, trans_state_to);
					if (trans_id == -1)
						log_trans_prob_top_down[to][from] = log_prob_no_trans;
					else
						log_trans_prob_top_down[to][from] = trans_model.GetTransitionLogProb_TopDown(trans_id);
				}
			}
			for (size_t from = 0; from < state_tol_num; from++) {
				for (size_t to = 0; to < state_tol_num; to++) {
					int32 trans_state_from = trans_model.PairToState(phone_id, from);
					int32 trans_state_to = trans_model.PairToState(phone_id, to);
					int32 trans_id = trans_model.StatePairToTransitionId_LeftRight(trans_state_from, trans_state_to);
					if (trans_id == -1)
						log_trans_prob_left_right[to][from] = log_prob_no_trans;
					else
						log_trans_prob_left_right[to][from] = trans_model.GetTransitionLogProb_LeftRight(trans_id);
				}
			}
			
			// Store trans_state for every state in this phone  
			std::vector<int32> trans_states_for_this_phone(state_tol_num);
			// Store pdf_id for every state in this phone  
			std::vector<int32> pdf_index_for_this_phone(state_tol_num);
			for (size_t i = 0; i < state_tol_num; i++) {
				trans_state = trans_model.PairToState(phone_id, static_cast<int32>(i)); //���phone�е�i��state��Ӧ��trans_stateֵ
				trans_states_for_this_phone[i] = trans_state;
				pdf_index_for_this_phone[i] = trans_model.TransitionStateToForwardPdf(trans_state);
			}

			int32 pdf_index;
			size_t which_block_row = 0, which_block_col = 0,
				which_block_index = which_block_row * block_col_num + which_block_col;

			//------------------------compute the first block------------------------//
			which_block_index = 0;
			for (size_t i = 0; i < state_tol_num; i++) {
				const DiagGmm &pdf = am_gmm.GetPdf(pdf_index_for_this_phone[i]);//pdf_index should be 0-based
				log_delta_matrix[which_block_index][i] = (log_pai_top_down[i] + log_pai_left_right[i]) / 2 +
					LogLikelihood_Temp(pdf, features.Row(static_cast<MatrixIndexT>(which_block_index)));
			}
			//------------compute the first row, from second col to the end in block map-----------//
			which_block_row = 0;
			for (which_block_col = 1; which_block_col < block_col_num; which_block_col++) {
				which_block_index = which_block_row * block_col_num + which_block_col;
				for (size_t i = 0; i < state_tol_num; i++) {
					// ���� log_delta_{left_block}{left_block_state} + log_trans{current_block_state | left_block_state}
					size_t max_index = -1;
					BaseFloat max_sum = -1000000, temp_sum=-1000000;
					for (size_t from = 0; from < state_tol_num; from++) {
						temp_sum= log_delta_matrix[which_block_index - 1][from]+ log_trans_prob_left_right[i][from];
						if (temp_sum > max_sum) {
							max_index = from;
							max_sum = temp_sum;
						}
					}
					most_like_state_left[which_block_index][i] = max_index;

					const DiagGmm &pdf = am_gmm.GetPdf(pdf_index_for_this_phone[i]);//pdf_index should be 0-based
					log_delta_matrix[which_block_index][i] = (log_pai_top_down[i] + max_sum) / 2 +
						LogLikelihood_Temp(pdf, features.Row(static_cast<MatrixIndexT>(which_block_index)));
				}
			}
			//------------compute the first col, from second row to the end in block map-----------//
			which_block_col = 0;
			for (which_block_row = 1; which_block_row < block_row_num; which_block_row++) {
				which_block_index = which_block_row * block_col_num + which_block_col;
				for (size_t i = 0; i < state_tol_num; i++) {
					// ���� log_delta_{top_block}{top_block_state} + log_trans{current_block_state | top_block_state}
					size_t max_index = -1;
					BaseFloat max_sum = -1000000, temp_sum = -1000000;
					for (size_t from = 0; from < state_tol_num; from++) {
						temp_sum = log_delta_matrix[which_block_index - block_col_num][from] + log_trans_prob_top_down[i][from];
						if (temp_sum > max_sum) {
							max_index = from;
							max_sum = temp_sum;
						}
					}
					most_like_state_top[which_block_index][i] = max_index;

					const DiagGmm &pdf = am_gmm.GetPdf(pdf_index_for_this_phone[i]);//pdf_index should be 0-based
					log_delta_matrix[which_block_index][i] = (log_pai_left_right[i] + max_sum) / 2 +
						LogLikelihood_Temp(pdf, features.Row(static_cast<MatrixIndexT>(which_block_index)));
				}
			}
			//------------------------compute left blocks------------------------//
			for (which_block_row = 1; which_block_row < block_row_num; which_block_row++) {
				for (which_block_col = 1; which_block_col < block_col_num; which_block_col++) {
					which_block_index = which_block_row * block_col_num + which_block_col;
					for (size_t i = 0; i < state_tol_num; i++) {
						// ���� log_delta_{left_block}{left_block_state} + log_trans{current_block_state | left_block_state}
						size_t max_index = -1;
						BaseFloat max_sum_left = -1000000, max_sum_top = -1000000, temp_sum = -1000000;
						for (size_t from = 0; from < state_tol_num; from++) {
							temp_sum = log_delta_matrix[which_block_index - 1][from] + log_trans_prob_left_right[i][from];
							if (temp_sum > max_sum_left) {
								max_index = from;
								max_sum_left = temp_sum;
							}
						}
						most_like_state_left[which_block_index][i] = max_index;
						// ���� log_delta_{top_block}{top_block_state} + log_trans{current_block_state | top_block_state}
						max_index = -1; max_sum_top = -1000000; temp_sum = -1000000;
						for (size_t from = 0; from < state_tol_num; from++) {
							temp_sum = log_delta_matrix[which_block_index - block_col_num][from] + log_trans_prob_top_down[i][from];
							if (temp_sum > max_sum_top) {
								max_index = from;
								max_sum_top = temp_sum;
							}
						}
						most_like_state_top[which_block_index][i] = max_index;

						const DiagGmm &pdf = am_gmm.GetPdf(pdf_index_for_this_phone[i]);//pdf_index should be 0-based
						log_delta_matrix[which_block_index][i] = (max_sum_left + max_sum_top) / 2 +
							LogLikelihood_Temp(pdf, features.Row(static_cast<MatrixIndexT>(which_block_index)));
					}
				}
			}
			//-------------------all temp matrix OK--------------------//

			std::vector<int32> blocks2indices(block_tol_num, 0);
			std::vector<int32> blocks2states(block_tol_num, 0);

			//-------------------compute the log-like of this sample--------------------//
			//-------------------compute the trans_state corresponding to last-row-last-col block--------------------//
			size_t max_index = -1;
			BaseFloat max_log_like = -1000000;
			for (size_t final_state = 0; final_state < state_tol_num; final_state++) {
				if (log_delta_matrix[block_tol_num-1][final_state] > max_log_like) {
					max_index = final_state;
					max_log_like = log_delta_matrix[block_tol_num - 1][final_state];
				}
			}
			blocks2indices[block_tol_num - 1] = max_index;
			blocks2states[block_tol_num - 1] = trans_states_for_this_phone[max_index];
			tot_like += max_log_like;
			frame_count += block_tol_num;
			//-------------------compute the trans_state corresponding to last-row-but-not-last-col blocks--------------------//
			which_block_row = block_row_num - 1;
			for (which_block_col = block_col_num - 2; which_block_col >= 0; which_block_col--) {
				which_block_index = which_block_row * block_col_num + which_block_col;
				blocks2indices[which_block_index] = most_like_state_left[which_block_index + 1][blocks2indices[which_block_index + 1]];
				blocks2states[which_block_index] = trans_states_for_this_phone[blocks2indices[which_block_index]];
			}
			//-------------------compute the trans_state corresponding to last-col-but-not-last-row blocks--------------------//
			which_block_col = block_col_num - 1;
			for (which_block_row = block_row_num - 2; which_block_row >= 0; which_block_row--) {
				which_block_index = which_block_row * block_col_num + which_block_col;
				blocks2indices[which_block_index] = most_like_state_top[which_block_index + block_col_num][blocks2indices[which_block_index + block_col_num]];
				blocks2states[which_block_index] = trans_states_for_this_phone[blocks2indices[which_block_index]];
			}
			//-------------------compute the trans_state corresponding to left blocks--------------------//
			for (which_block_row = block_row_num - 2; which_block_row >= 0; which_block_row--) {
				for (which_block_col = block_col_num - 2; which_block_col >= 0; which_block_col--) {
					which_block_index = which_block_row * block_col_num + which_block_col;
					BaseFloat delta_down= log_delta_matrix[which_block_index][most_like_state_top[which_block_index + block_col_num][blocks2indices[which_block_index + block_col_num]]],
						delta_right = log_delta_matrix[which_block_index][most_like_state_left[which_block_index + 1][blocks2indices[which_block_index + 1]]];
					if (delta_down > delta_right) {
						blocks2indices[which_block_index] = most_like_state_top[which_block_index + block_col_num][blocks2indices[which_block_index + block_col_num]];
						blocks2states[which_block_index] = trans_states_for_this_phone[blocks2indices[which_block_index]];
					}
					else {
						blocks2indices[which_block_index] = most_like_state_left[which_block_index + 1][blocks2indices[which_block_index + 1]];
						blocks2states[which_block_index] = trans_states_for_this_phone[blocks2indices[which_block_index]];
					}
				}
			}

			alignment_writer.Write(utt, blocks2states);
			num_done++;
		}
		KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like / frame_count)
			<< " over " << frame_count << " frames.";
		KALDI_LOG << "Done " << num_done << ", errors on " << num_err;
		return (num_done != 0 ? 0 : 1);
	}
	catch (const std::exception &e) {
		std::cerr << e.what();
		return -1;
	}
}


