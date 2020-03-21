// gmmbin/gmm-acc-stats-ali.cc

// Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)

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
#include "gmm/mle-am-diag-gmm.h"

int main(int argc, char *argv[]) {
	using namespace kaldi;
	typedef kaldi::int32 int32;
	try {
		const char *usage =
			"Accumulate stats for GMM training.\n"
			"Usage:  gmm-acc-stats-ali-2D [options] <model-in> <feature-rspecifier> "
			"<block-rspecifier> <alignments-rspecifier> <stats-out>\n"
			"e.g.:\n gmm-acc-stats-ali 1.mdl scp:train.scp ark:block ark:1.ali 1.acc\n";

		ParseOptions po(usage); // ʹ��usage�ִ���ʼ��һ��ParseOptions���ʵ��po
		bool binary = true;
		po.Register("binary", &binary, "Write output in binary mode");// ��ParseOptions����ע��������ѡ��(Option�Ľṹ�������Լ���ע�ắ��)
		po.Read(argc, argv); // �������в������н���
		// ����Ƿ������Ч������λ�ò���
		if (po.NumArgs() != 5) {
			po.PrintUsage();
			exit(1);
		}
		// ��ȡָ��λ�õ������в���������ֵ����Ӧ��ѡ��
		std::string model_filename = po.GetArg(1),
			feature_rspecifier = po.GetArg(2),
			block_rspecifier = po.GetArg(3),
			alignments_rspecifier = po.GetArg(4),
			accs_wxfilename = po.GetArg(5);

		AmDiagGmm am_gmm;
		TransitionModel_2D trans_model;
		{
			bool binary; // binary��ֵͨ��kiʵ������������ж���������Ϊ1���ı���ʽΪ0
			Input ki(model_filename, &binary); // Kaldi���ݿ�ͷ�Ƿ�"\0B"���ж��Ƕ����ƻ����ı���ʽ��ͬʱ׼����һ����ki
			trans_model.Read(ki.Stream(), binary);// ��.mdl�����ȡ����TransitionModel��Ķ���trans_model
			am_gmm.Read(ki.Stream(), binary);// ��.mdl�����ȡ����AmDiagGmm��Ķ���am_gmm(����NUMPDFS��DiagGmms���AmDiagGmm)
		}

		Vector<double> transition_accs_top_down; // to save weighted counts of top2down trans_id
		Vector<double> transition_accs_left_right; // to save weighted counts of left2right trans_id
		trans_model.InitStats_TopDown(&transition_accs_top_down); // ��transition_accs_top_down�ĳ��ȳ�ʼ��Ϊ����ֱת�ƻ���+1
		trans_model.InitStats_LeftRight(&transition_accs_left_right); // ��transition_accs_top_down�ĳ��ȳ�ʼ��Ϊ����ֱת�ƻ���+1
		AccumAmDiagGmm gmm_accs; // AccumAmDiagGmm���е�˽�б���Ϊvector<AccumDiagGmm*>��AccumDiagGmm������һ��DiagGmm���������������Ϣ
		gmm_accs.Init(am_gmm, kGmmAll); // ʹ��am_gmm����ʼ��ÿ��GMM���ۻ���gmm_accs��kGmmAll��0x00F��ָ���¾�ֵ�����Ȩ�ء�trans

		double tot_like = 0.0;
		kaldi::int64 tot_t = 0;

		SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
		RandomAccessInt32VectorReader alignments_reader(alignments_rspecifier);
		RandomAccessInt32VectorReader block_info_reader(block_rspecifier);

		int32 num_done = 0, num_err = 0;
		for (; !feature_reader.Done(); feature_reader.Next()) { // ����ÿ���������е�����ѵ������
			std::string key = feature_reader.Key(); // �����������
			if (!alignments_reader.HasKey(key)) {
				KALDI_WARN << "No alignment for utterance " << key;
				num_err++;
				continue;
			}
			else if(!block_info_reader.HasKey(key)) {
				KALDI_WARN << "No block information for utterance " << key;
				num_err++;
				continue;
			}
			else {
				const Matrix<BaseFloat> &mat = feature_reader.Value(); // mat�洢��ǰfeature_readerָ���Value����ǰ������feats��
				const std::vector<int32> &alignment = alignments_reader.Value(key); // alignment�洢alignments_reader��keyָ���Ķ�����Ϣ
				const std::vector<int32> &block_info = block_info_reader.Value(key); // blcok_info�洢block_info_reader��keyָ����֡����Ϣ
				if (alignment.size() != mat.NumRows()) { // ͨ�����feats֡���Ͷ������Ƿ�������ж������Ƿ���Ч
					KALDI_WARN << "Alignments has wrong size " << (alignment.size())
						<< " vs. " << (mat.NumRows());
					num_err++;
					continue;
				}
				// ���ĳ��sample��block-information����������row_num col_num tol_num�����������+1���������sample
				if (block_info.size() != 3) {
					KALDI_WARN << "Block information size supposed to be 3, but get "
						<< block_info.size() << " from " << key << " instead.";
					num_err++;
					continue;
				}
				size_t block_row_num = static_cast<size_t>(block_info[0]),
					block_col_num = static_cast<size_t>(block_info[1]),
					block_tol_num = static_cast<size_t>(block_info[2]);
				if (block_row_num*block_col_num != block_tol_num) { // ���block_info����ȷ��
					KALDI_WARN << "Block rows number * cols number not equal to total number, sample name:" << key
						<< "rows:" << block_row_num << " cols:" << block_col_num << " total:" << block_tol_num;
					num_err++;
					continue;
				}
				if (alignment.size() != block_tol_num) { // ���block_info����ȷ��
					KALDI_WARN << "Block total number has error " << block_tol_num
						<< " vs. Alignment size " << alignment.size();
					num_err++;
					continue;
				}
				num_done++;
				BaseFloat tot_like_this_file = 0.0;

				// ���ڵ�ǰ�ļ��е�ÿһ֡����������pdf_id��Ӧ��������Ŀ����ֵ��trans_id����ֵ���ļ����ɸ���
				for (size_t i = 0; i < alignment.size(); i++) {
					int32 this_state = alignment[i],
						down_side_state = (i + block_col_num) >= block_tol_num ? 0 : alignment[i + block_col_num],
						right_side_state = ((i + 1) % block_col_num) == 0 ? 0 : alignment[i + 1];
					int32 tid_top_down = trans_model.StatePairToTransitionId_TopDown(this_state, down_side_state),  // transition identifier.
						tid_left_right = trans_model.StatePairToTransitionId_LeftRight(this_state, right_side_state),  // transition identifier.
						pdf_id = trans_model.TransitionIdToPdf_TopDown(tid_top_down); // ͨ��TM��id2pdf_id_ӳ���ϵ�ҵ���Ӧ��pdf_id
					if (tid_top_down == -1) {
						KALDI_WARN << "Alignment error, TopDown transition from trans_state " << this_state
							<< " to " << down_side_state << " is not allowed!";
						num_err++;
						continue;
					}
					if (tid_left_right == -1) {
						KALDI_WARN << "Alignment error, LeftRight transition from trans_state " << this_state
							<< " to " << right_side_state << " is not allowed!";
						num_err++;
						continue;
					}
					trans_model.Accumulate_TopDown(1.0, tid_top_down, &transition_accs_top_down); // ��transition_accs�ж�Ӧ�Ļ���trans_id��λ�ü���ֵ��һ
					trans_model.Accumulate_LeftRight(1.0, tid_left_right, &transition_accs_left_right); // ��transition_accs�ж�Ӧ�Ļ���trans_id��λ�ü���ֵ��һ
				   // am_gmm���.mdl��GMMs����ز�����mat��ŵ�ǰ�ļ�����������ÿ��Ϊһ��������������pdf_id���ɶ������и����ĵ�ǰ֡��Ӧ��GMM
					tot_like_this_file += gmm_accs.AccumulateForGmm(am_gmm, mat.Row(i), // �ۼ�ÿһ֡�ɶ�Ӧ��GMM�����Ķ�������
						pdf_id, 1.0);
				}
				tot_like += tot_like_this_file; // tot_like_this_file��ŵ�ǰ�ļ���feature vectors���ɶ�Ӧ�Ķ������У�GMM sequence�������Ķ�������֮��
				tot_t += alignment.size(); // tot_t�ۼ������ļ�����֡��
				if (num_done % 50 == 0) { // ÿ����50���ļ����һ��log��Ϣ��������50*n���ļ���ƽ��ÿ֡���ɸ��ʣ������ڵ�ǰ�Ķ�����Ϣ��
					KALDI_LOG << "Processed " << num_done << " utterances; for utterance "
						<< key << " avg. like is "
						<< (tot_like_this_file / alignment.size())
						<< " over " << alignment.size() << " frames.";
				}
			}
		}
		KALDI_LOG << "Done " << num_done << " files, " << num_err
			<< " with errors.";

		// �����ļ�ÿһ֡�����ɸ��ʶ����ͣ�tot_like��/�����ļ���֡����tot_t��=ƽ��ÿ֡���ɸ��ʣ�avg like per frame��
		KALDI_LOG << "Overall avg like per frame (Gaussian only) = "
			<< (tot_like / tot_t) << " over " << tot_t << " frames.";

		{
			Output ko(accs_wxfilename, binary); // ��accs_wxfilenameΪ�ļ���ע������ļ����ko����ʽΪbinary��Ĭ��Ϊtrue��������������ָ����
			transition_accs_top_down.Write(ko.Stream(), binary); // �����л��������ļ�����frame�г��ֵļ���ֵtransition_accs��double��������д��accs_wxfilename
			transition_accs_left_right.Write(ko.Stream(), binary); // �����л��������ļ�����frame�г��ֵļ���ֵtransition_accs��double��������д��accs_wxfilename
			gmm_accs.Write(ko.Stream(), binary); // ��ÿ��DiagGmm��������Ĳ���gmm_accsд��accs_wxfilename
		}
		KALDI_LOG << "Written accs.";
		if (num_done != 0)
			return 0;
		else
			return 1;
	}
	catch (const std::exception &e) {
		std::cerr << e.what();
		return -1;
	}
}


